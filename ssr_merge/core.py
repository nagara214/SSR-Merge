"""SSR core algorithm: hook + accumulate G/Q + solve R + reparameterize B.

This module is framework-agnostic. It only depends on ``torch`` and operates
purely on tensor inputs. The backbone-specific glue (loading a pipeline,
locating Linear modules, running calibration) lives in :mod:`runners`.

Reference: SSR-Merge: Subspace Signal Routing for Training-Free LoRA
Merging in Diffusion Models (ICML 2026).

Notation per layer (one entry below corresponds to one section of the
paper; see Sec. 3 for derivations):
    * K task-specific LoRA pairs ``(A_k, B_k)`` with rank ``r_k`` (the
      paper assumes a common rank; this implementation allows per-task
      ``r_k`` to differ).
    * ``A_comb = concat([A_k], dim=0)`` shape ``(sum_k r_k, in_dim)``.
    * ``B_comb = concat([B_k], dim=1)`` shape ``(out_dim, sum_k r_k)``.
    * For each calibration token ``x`` we form ``z = A_comb @ x``, with
      ``z_k`` the slice of ``z`` belonging to task ``k`` (equal to
      ``A_k @ x``, i.e. ``H_k`` in the paper). We then accumulate:
          ``G += z z^T``                  (correlation matrix)
          ``Q[k_slice, :] += z_k z^T``    (directional guide, k-th block)
      Summed across all tokens this gives the paper's
      ``G = sum_k Z_k Z_k^T`` and ``Q_k = H_k Z_k^T``.
    * Solve the router ``R = Q G^{-1}`` (in float64, via
      ``torch.linalg.solve`` for numerical stability), then reparameterize:
          ``A_merged = A_comb``
          ``B_merged = B_comb @ R``    (equivalently ``B_comb_tilde``)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

log = logging.getLogger("ssr_merge.core")

LayerName = str
LoraDict = Dict[LayerName, Tuple[Tensor, Tensor]]  # {layer: (A, B)}


class SSRMerger:
    """Subspace Signal Routing merger.

    The merger ingests a list of K per-task LoRA dictionaries, registers
    forward hooks on the corresponding base-model Linear modules, accumulates
    the second-order statistics ``G`` and ``Q`` during one (or a few)
    calibration forward passes, and finally solves the optimal router ``R``
    and reparameterizes it into the merged LoRA weights.

    Per-layer behaviour
    -------------------
    Each layer is treated independently. If only ``K'`` of the ``K`` tasks
    contribute a LoRA on a given layer, SSR is applied only over those
    ``K'`` tasks. The remaining tasks are ignored on that layer.

    * ``K' >= 2``: standard SSR merging over the active subset.
    * ``K' == 1``: a single LoRA contributes; passed through as-is
      (equivalent to identity routing).
    * ``K' == 0``: layer is skipped, base model is unchanged on that layer.

    LoRA ranks may differ across tasks (and across layers); the algorithm
    handles this naturally via ``A_comb = cat(..., dim=0)`` and a
    per-task block index ``r_dims``.
    """

    def __init__(
        self,
        loras_per_task: Sequence[LoraDict],
        lambda_reg: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        if len(loras_per_task) < 2:
            raise ValueError(
                f"SSR merging requires at least 2 LoRAs (got K={len(loras_per_task)})."
            )
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be >= 0 (got {lambda_reg}).")

        self.loras_per_task = [dict(d) for d in loras_per_task]
        self.K = len(self.loras_per_task)
        self.lambda_reg = float(lambda_reg)
        self.stats_device = device

        # Per-layer registry; only layers covered by at least 1 task are listed.
        # layer_stats[name] = {
        #   "active_tasks": [k indices that have a LoRA on this layer],
        #   "A_list": [A_k for k in active_tasks],
        #   "B_list": [B_k for k in active_tasks],
        #   "r_dims": [r_k for k in active_tasks],
        #   "A_comb": (sum_k' r_k, in_dim),
        #   "G": (R, R), "Q": (R, R) accumulators on `device`,
        #   "count": number of samples accumulated,
        # }
        self.layer_stats: Dict[LayerName, dict] = {}
        self._build_layer_registry()

        # Filled by :meth:`attach`.
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._attached_modules: Dict[LayerName, nn.Module] = {}
        self._dead_tasks: set = set()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _build_layer_registry(self) -> None:
        all_layers: set = set()
        for d in self.loras_per_task:
            all_layers.update(d.keys())

        if not all_layers:
            raise RuntimeError(
                "No LoRA layers found across the K inputs; "
                "are they trained on the same base model?"
            )

        for name in sorted(all_layers):
            active_tasks: List[int] = []
            A_list: List[Tensor] = []
            B_list: List[Tensor] = []
            for k, d in enumerate(self.loras_per_task):
                if name not in d:
                    continue
                A, B = d[name]
                if A.dim() != 2 or B.dim() != 2:
                    raise ValueError(
                        f"Layer '{name}' task {k}: expected 2D A and B, "
                        f"got A.dim={A.dim()}, B.dim={B.dim()}."
                    )
                if A.shape[0] != B.shape[1]:
                    raise ValueError(
                        f"Layer '{name}' task {k}: rank mismatch between A and B "
                        f"(A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)})."
                    )
                active_tasks.append(k)
                A_list.append(A.to(torch.float32))
                B_list.append(B.to(torch.float32))

            if len(active_tasks) == 0:
                # Should not happen since name in all_layers, but be defensive.
                continue

            if len(active_tasks) == 1:
                # K' == 1: passthrough, no statistics needed.
                self.layer_stats[name] = {
                    "active_tasks": active_tasks,
                    "A_list": A_list,
                    "B_list": B_list,
                    "passthrough": True,
                }
                log.info(
                    "Layer '%s': only 1 LoRA contributes (task %d); passing through.",
                    name,
                    active_tasks[0],
                )
                continue

            # K' >= 2: prepare for SSR.
            A_comb = torch.cat(A_list, dim=0)
            total_rank = A_comb.shape[0]
            r_dims = [a.shape[0] for a in A_list]

            self.layer_stats[name] = {
                "active_tasks": active_tasks,
                "A_list": A_list,
                "B_list": B_list,
                "r_dims": r_dims,
                "A_comb": A_comb,
                "G": torch.zeros((total_rank, total_rank), device=self.stats_device),
                "Q": torch.zeros((total_rank, total_rank), device=self.stats_device),
                "count": 0,
                "passthrough": False,
            }
            if len(active_tasks) < self.K:
                missing = [k for k in range(self.K) if k not in active_tasks]
                log.info(
                    "Layer '%s': %d/%d tasks contribute (missing in tasks %s).",
                    name,
                    len(active_tasks),
                    self.K,
                    missing,
                )

    # ------------------------------------------------------------------ #
    # Hook management
    # ------------------------------------------------------------------ #
    def attach(self, named_modules: Dict[LayerName, nn.Module]) -> None:
        """Register forward hooks on the provided Linear modules.

        Parameters
        ----------
        named_modules
            Mapping ``{layer_name: nn.Module}``. The keys must use the same
            namespace as the keys in the LoRA dicts passed to ``__init__``.
            Modules for layers in passthrough or skip state are silently
            ignored (no hook is registered).
        """
        for name, module in named_modules.items():
            stats = self.layer_stats.get(name)
            if stats is None or stats.get("passthrough", False):
                continue
            module._ssr_layer_name = name  # type: ignore[attr-defined]
            handle = module.register_forward_hook(self._make_hook())
            self._hooks.append(handle)
            self._attached_modules[name] = module

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for name, module in self._attached_modules.items():
            if hasattr(module, "_ssr_layer_name"):
                delattr(module, "_ssr_layer_name")
        self._attached_modules.clear()

    def set_active_task(self, task_idx: int) -> None:
        """Set which task index the next calibration forward corresponds to.

        SSR accumulates statistics one task at a time: the user is expected
        to call this with ``task_idx = 0, 1, ..., K-1`` and run a calibration
        forward (with the matching LoRA loaded into the base model) between
        each call. Tasks previously marked as failed are silently ignored.
        """
        if task_idx < 0 or task_idx >= self.K:
            raise IndexError(f"task_idx={task_idx} out of range [0, {self.K}).")
        if task_idx in self._dead_tasks:
            log.warning("Task %d was marked failed; calibration will be a no-op.", task_idx)
        self._current_task = task_idx

    def mark_task_failed(self, task_idx: int) -> None:
        """Mark a task as failed (e.g. calibration OOM) and drop it from the merge.

        The failed task's contributions are zeroed out in every layer's
        ``G`` and ``Q`` accumulators (in case the failure happened
        mid-calibration after some hooks had already fired). Already-collected
        statistics for *other* tasks are preserved.

        After solve, the failed task's block in the router ``R`` is naturally
        zero (because ``G_dead = 0``, ``Q_dead = 0``, and the ridge term
        ``lambda I`` makes the system well-defined), so the failed task does
        not contribute to the merged LoRA. The active_tasks list is updated
        so that no further hook firings can accumulate into the failed task's
        block.
        """
        if task_idx < 0 or task_idx >= self.K:
            raise IndexError(f"task_idx={task_idx} out of range [0, {self.K}).")
        log.warning("Dropping task %d from merge due to upstream failure.", task_idx)

        for stats in self.layer_stats.values():
            if task_idx not in stats["active_tasks"]:
                continue
            local_idx = stats["active_tasks"].index(task_idx)
            if stats.get("passthrough", False):
                stats["active_tasks"].remove(task_idx)
                stats["A_list"].pop(local_idx)
                stats["B_list"].pop(local_idx)
                continue
            r_dims = stats["r_dims"]
            start = sum(r_dims[:local_idx])
            end = start + r_dims[local_idx]
            stats["G"][start:end, :] = 0
            stats["G"][:, start:end] = 0
            stats["Q"][start:end, :] = 0
            stats["Q"][:, start:end] = 0
            stats["B_list"][local_idx] = torch.zeros_like(stats["B_list"][local_idx])

        self._dead_tasks.add(task_idx)

        n_surviving = self.K - len(self._dead_tasks)
        if n_surviving < 2:
            raise RuntimeError(
                f"After dropping failed task {task_idx}, only {n_surviving} task(s) "
                f"survive; need at least 2 to perform SSR merging."
            )

    def _make_hook(self) -> Callable:
        def hook_fn(module, inputs, _outputs):
            name = getattr(module, "_ssr_layer_name", None)
            if name is None:
                return
            stats = self.layer_stats.get(name)
            if stats is None or stats.get("passthrough", False):
                return
            task_idx = getattr(self, "_current_task", None)
            if task_idx is None or task_idx in self._dead_tasks:
                return
            active_tasks = stats["active_tasks"]
            if task_idx not in active_tasks:
                return  # this task does not contribute on this layer
            local_idx = active_tasks.index(task_idx)

            with torch.no_grad():
                A_comb = stats["A_comb"]
                in_dim = A_comb.shape[1]
                x = inputs[0].detach()
                if x.shape[-1] != in_dim:
                    return
                x = x.reshape(-1, in_dim).t().to(torch.float32)  # (in_dim, N)
                A_comb_dev = A_comb.to(x.device)
                z = torch.mm(A_comb_dev, x)  # (R, N)

                delta_G = torch.mm(z, z.t())
                stats["G"] += delta_G.to(self.stats_device)

                r_dims = stats["r_dims"]
                start = sum(r_dims[:local_idx])
                end = start + r_dims[local_idx]
                z_self = z[start:end, :]
                delta_Q = torch.mm(z_self, z.t())
                stats["Q"][start:end, :] += delta_Q.to(self.stats_device)
                stats["count"] += x.shape[1]

        return hook_fn

    # ------------------------------------------------------------------ #
    # Solve / reparameterize
    # ------------------------------------------------------------------ #
    def solve(self) -> Dict[LayerName, Tuple[Tensor, Tensor]]:
        """Solve the per-layer router R and return the merged LoRA dict.

        Returns
        -------
        dict
            ``{layer_name: (A_merged, B_merged)}``. ``A_merged`` is the
            concatenated ``A_comb`` (or the single passthrough ``A`` when
            ``K' == 1``); ``B_merged`` is ``B_comb @ R`` (or the single
            passthrough ``B`` when ``K' == 1``).
        """
        merged: Dict[LayerName, Tuple[Tensor, Tensor]] = {}
        n_passthrough = 0
        n_solve_failed = 0
        n_no_activation = 0

        for name, stats in self.layer_stats.items():
            if stats.get("passthrough", False):
                A = stats["A_list"][0]
                B = stats["B_list"][0]
                merged[name] = (A, B)
                n_passthrough += 1
                continue

            N = stats["count"]
            A_comb = stats["A_comb"]
            B_comb = torch.cat(stats["B_list"], dim=1)
            R_dim = A_comb.shape[0]

            if N == 0:
                log.warning(
                    "Layer '%s' received no activations during calibration; "
                    "falling back to identity router R=I.",
                    name,
                )
                R = torch.eye(R_dim, dtype=torch.float32)
                n_no_activation += 1
            else:
                G = (stats["G"] / N) + torch.eye(R_dim, device=stats["G"].device) * self.lambda_reg
                Q = stats["Q"] / N
                try:
                    G64 = G.to(torch.float64)
                    Q64 = Q.to(torch.float64)
                    R = torch.linalg.solve(G64, Q64.t()).t().to(torch.float32)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "Solve failed for layer '%s' (%s); falling back to identity R=I.",
                        name,
                        e,
                    )
                    R = torch.eye(R_dim, dtype=torch.float32)
                    n_solve_failed += 1

            B_merged = torch.mm(B_comb, R)
            merged[name] = (A_comb.contiguous(), B_merged.contiguous())

        log.info(
            "Solved %d layers (%d passthrough, %d solve-fallback, %d no-activation-fallback).",
            len(self.layer_stats),
            n_passthrough,
            n_solve_failed,
            n_no_activation,
        )
        return merged

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[LayerName, dict]:
        """Return a shallow snapshot of internal per-layer statistics (for debug)."""
        out: Dict[LayerName, dict] = {}
        for name, stats in self.layer_stats.items():
            entry = {
                "active_tasks": list(stats["active_tasks"]),
                "passthrough": bool(stats.get("passthrough", False)),
            }
            if not entry["passthrough"]:
                entry["count"] = int(stats["count"])
                entry["r_dims"] = list(stats["r_dims"])
                entry["A_comb_shape"] = tuple(stats["A_comb"].shape)
                entry["G_shape"] = tuple(stats["G"].shape)
            out[name] = entry
        return out
