"""Generic SSR calibration loop.

This module contains a single function, :func:`run`, that drives SSR
calibration on any diffsynth-style pipeline. A pipeline is "diffsynth-style"
if it exposes:

* a transformer attribute (default name: ``dit``);
* ``pipe.load_lora(transformer, lora_path, alpha=...)``;
* ``pipe(prompt=..., seed=..., num_inference_steps=...)`` for generation.

The LoRA loader must expose ``loader.convert_state_dict(raw_state_dict)``
returning a flat dict keyed by ``<layer>.lora_A.weight`` /
``<layer>.lora_B.weight`` matching the transformer's module names. Diffsynth's
``GeneralLoRALoader`` and ``FluxLoRALoader`` both satisfy this.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .core import SSRMerger

log = logging.getLogger("ssr_merge.runners")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _resolve_lora_path(path_or_repo: str) -> str:
    """Accept a local path or an ``org/repo[:filename]`` HuggingFace spec."""
    if os.path.exists(path_or_repo):
        return path_or_repo
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise FileNotFoundError(
            f"LoRA path not found locally and huggingface_hub is not installed: {path_or_repo}"
        ) from e

    if ":" in path_or_repo:
        repo_id, filename = path_or_repo.split(":", 1)
        return hf_hub_download(repo_id=repo_id, filename=filename)
    repo_dir = snapshot_download(repo_id=path_or_repo)
    candidates = [f for f in os.listdir(repo_dir) if f.endswith(".safetensors")]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot infer LoRA file from repo '{path_or_repo}': "
            f"found {len(candidates)} safetensors files, expected exactly 1. "
            f"Use 'org/repo:filename' to disambiguate."
        )
    return os.path.join(repo_dir, candidates[0])


def _validate_inputs(loras: Sequence[str], prompts: Sequence[str]) -> None:
    if len(loras) < 2:
        raise ValueError(
            f"SSR merging requires at least 2 LoRAs (got K={len(loras)})."
        )
    if len(prompts) != len(loras):
        raise ValueError(
            f"Number of prompts ({len(prompts)}) must match number of LoRAs "
            f"({len(loras)})."
        )
    seen: Dict[str, int] = {}
    for i, p in enumerate(loras):
        if p in seen:
            log.warning("Duplicate LoRA paths at indices %d and %d (%s).", seen[p], i, p)
        seen[p] = i


def _extract_lora_pairs(state_dict: Dict[str, Tensor]) -> Dict[str, Tuple[Tensor, Tensor]]:
    """Convert a flat LoRA state dict to ``{layer_name: (A, B)}``."""
    pairs: Dict[str, Tuple[Tensor, Tensor]] = {}
    for key in state_dict:
        if not key.endswith(".lora_A.weight"):
            continue
        layer = key[: -len(".lora_A.weight")]
        b_key = layer + ".lora_B.weight"
        if b_key not in state_dict:
            log.info("Layer '%s' has lora_A but no lora_B; skipping.", layer)
            continue
        pairs[layer] = (state_dict[key], state_dict[b_key])
    return pairs


def _emit_lora_safetensors(
    merged: Dict[str, Tuple[Tensor, Tensor]],
    out_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Tensor]:
    """Flatten ``{layer: (A, B)}`` to a safetensors-ready dict (diffsynth keys)."""
    out: Dict[str, Tensor] = {}
    for layer, (A, B) in merged.items():
        out[f"{layer}.lora_A.weight"] = A.contiguous().to(out_dtype)
        out[f"{layer}.lora_B.weight"] = B.contiguous().to(out_dtype)
        out[f"{layer}.lora_A.alpha"] = torch.tensor(A.shape[0], dtype=torch.float32)
    return out


# ---------------------------------------------------------------------- #
# Generic runner
# ---------------------------------------------------------------------- #
def run(
    pipe: Any,
    loader: Any,
    loras: Sequence[str],
    prompts: Sequence[str],
    output: str | None = None,
    lambda_reg: float = 1e-4,
    calib_steps: int = 1,
    seed: int = 42,
    transformer_attr: str = "dit",
    out_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Tensor]:
    """Run SSR calibration on a pre-built diffsynth-style pipeline.

    Parameters
    ----------
    pipe
        A constructed pipeline that exposes ``getattr(pipe, transformer_attr)``,
        ``pipe.load_lora(transformer, lora_path, alpha=1.0)``, and
        ``pipe(prompt=..., seed=..., num_inference_steps=...)``. Any diffsynth
        image / video pipeline satisfies this.
    loader
        A LoRA loader with a ``convert_state_dict(raw)`` method returning a
        flat dict keyed by ``<layer>.lora_A.weight`` / ``<layer>.lora_B.weight``
        matching the transformer's module names. Diffsynth's
        ``GeneralLoRALoader`` works for most pipelines; FLUX has its own
        ``FluxLoRALoader``.
    loras
        K LoRA paths or HuggingFace ``org/repo[:filename]`` specs (K >= 2).
    prompts
        K calibration prompts, one per LoRA in matching order.
    output
        If given, save the merged LoRA to this path (must end with
        ``.safetensors``) in addition to returning the dict.
    lambda_reg
        Ridge regularizer on ``G``. Default 1e-4.
    calib_steps
        Number of diffusion steps per calibration forward. Default 1.
    seed
        Calibration RNG seed.
    transformer_attr
        Attribute name of the transformer module on ``pipe``. Default ``"dit"``.
    out_dtype
        Dtype of the emitted LoRA weights (the SSR statistics are always
        accumulated in float32).

    Returns
    -------
    dict
        Flat safetensors-ready state dict for the merged LoRA.
    """
    if output is not None and not output.endswith(".safetensors"):
        raise ValueError(f"output must end with .safetensors (got {output!r}).")
    _validate_inputs(loras, prompts)
    if calib_steps < 1:
        raise ValueError(f"calib_steps must be >= 1 (got {calib_steps}).")

    transformer = getattr(pipe, transformer_attr, None)
    if transformer is None:
        raise AttributeError(
            f"Pipeline of type {type(pipe).__name__} has no attribute "
            f"'{transformer_attr}'. Pass transformer_attr=... to override."
        )

    resolved_paths: List[str] = [_resolve_lora_path(p) for p in loras]

    # Load + convert all LoRAs to the pipeline's key namespace.
    loras_per_task: List[Dict[str, Tuple[Tensor, Tensor]]] = []
    for path in resolved_paths:
        raw = load_file(path)
        converted = loader.convert_state_dict(raw)
        loras_per_task.append(_extract_lora_pairs(converted))

    merger = SSRMerger(loras_per_task, lambda_reg=lambda_reg, device="cpu")
    log.info(
        "Built SSRMerger over %d tasks; %d layers in the merge plan.",
        len(loras_per_task),
        len(merger.layer_stats),
    )

    # Snapshot base transformer weights so we can restore between tasks.
    original_state = {k: v.detach().cpu().clone() for k, v in transformer.state_dict().items()}

    def _restore() -> None:
        transformer.load_state_dict(original_state)
        if hasattr(pipe, "clear_lora"):
            try:
                pipe.clear_lora()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for i, (path, prompt) in enumerate(zip(resolved_paths, prompts)):
        log.info("[%d/%d] Calibrating with prompt: %r", i + 1, len(prompts), prompt)
        t0 = time.time()
        try:
            _restore()
            pipe.load_lora(transformer, path, alpha=1.0)

            named_modules: Dict[str, torch.nn.Module] = {}
            for name, mod in transformer.named_modules():
                if isinstance(mod, torch.nn.Linear) and name in merger.layer_stats:
                    named_modules[name] = mod
            merger.set_active_task(i)
            merger.attach(named_modules)

            with torch.no_grad():
                pipe(prompt=prompt, seed=seed, num_inference_steps=calib_steps)

            merger.detach()
            log.info("[%d/%d] Calibration done in %.1fs.", i + 1, len(prompts), time.time() - t0)
        except Exception as e:  # noqa: BLE001
            log.warning("Calibration failed for task %d (%r): %s", i, prompt, e)
            merger.detach()
            merger.mark_task_failed(i)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _restore()

    log.info("Solving routers and reparameterizing merged LoRA...")
    merged = merger.solve()
    merged_sd = _emit_lora_safetensors(merged, out_dtype=out_dtype)

    if output is not None:
        out_dir = os.path.dirname(os.path.abspath(output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_file(merged_sd, output)
        log.info("Merged LoRA saved to %s", os.path.abspath(output))

    return merged_sd
