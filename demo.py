"""SSR-merge demo on a DiT-based backbone.

Usage::

    # Reproduce the paper's cat + dog merge on FLUX.1-dev (default).
    # The two DreamBooth LoRAs are downloaded from Google Drive on first run.
    python demo.py

    # Use a different backbone with your own LoRAs:
    python demo.py --backbone qwen \\
        --loras task_a.safetensors task_b.safetensors \\
        --prompts "a photo of ..." "a photo of ..."

Supported backbones: flux, qwen, z_image, hidream, flux2.

For any other diffsynth pipeline, just write a ``build_<name>`` function
following the same recipe (construct the pipeline + a LoRA loader) and
add it to ``BUILDERS``. The rest of this script is bookkeeping.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, Dict, Tuple

import torch

from ssr_merge import run

# The paper's DreamBooth cat / dog LoRAs, hosted on Google Drive.
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1riatiorE8WYgKGUgc8ZwARv_O37Ui_hu"
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_loras")
DEFAULT_LORAS = [
    os.path.join(DEMO_DIR, "flux_lora_cat.safetensors"),
    os.path.join(DEMO_DIR, "flux_lora_dog.safetensors"),
]
# Both LoRAs were trained with the DreamBooth identifier "sks".
DEFAULT_PROMPTS = ["a sks cat", "a sks dog"]

_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def ensure_demo_loras() -> None:
    """Download the cat / dog demo LoRAs from Google Drive if not present."""
    if all(os.path.exists(p) for p in DEFAULT_LORAS):
        return
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "The demo LoRAs are hosted on Google Drive; please install gdown:\n"
            "    pip install gdown\n"
            f"or download them manually from {GDRIVE_FOLDER} into {DEMO_DIR}/"
        ) from e
    os.makedirs(DEMO_DIR, exist_ok=True)
    logging.info("Downloading demo LoRAs from Google Drive into %s ...", DEMO_DIR)
    gdown.download_folder(GDRIVE_FOLDER, output=DEMO_DIR, quiet=False, use_cookies=False)


def _device_dtype(device: str, dtype: str) -> Tuple[str, torch.dtype]:
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA unavailable; falling back to CPU.")
        device = "cpu"
    return device, _DTYPE[dtype]


# ---------------------------------------------------------------------- #
# Backbone builders
# Each returns (pipe, loader). Pipelines are constructed exactly as in the
# corresponding `examples/<backbone>/model_inference/*.py` from DiffSynth.
# ---------------------------------------------------------------------- #
def build_flux(device: str, dtype: str):
    from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
    from diffsynth.utils.lora.flux import FluxLoRALoader

    device, dt = _device_dtype(device, dtype)
    base = "black-forest-labs/FLUX.1-dev"
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=dt, device=device,
        model_configs=[
            ModelConfig(model_id=base, origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="text_encoder_2/*.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="ae.safetensors"),
        ],
    )
    return pipe, FluxLoRALoader(device="cpu")


def build_qwen(device: str, dtype: str):
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
    from diffsynth.utils.lora.general import GeneralLoRALoader

    device, dt = _device_dtype(device, dtype)
    base = "Qwen/Qwen-Image"
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=dt, device=device,
        model_configs=[
            ModelConfig(model_id=base, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=base, origin_file_pattern="tokenizer/"),
    )
    return pipe, GeneralLoRALoader(device="cpu")


def build_z_image(device: str, dtype: str):
    from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
    from diffsynth.utils.lora.general import GeneralLoRALoader

    device, dt = _device_dtype(device, dtype)
    base = "Tongyi-MAI/Z-Image-Turbo"
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=dt, device=device,
        model_configs=[
            ModelConfig(model_id=base, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=base, origin_file_pattern="tokenizer/"),
    )
    return pipe, GeneralLoRALoader(device="cpu")


def build_hidream(device: str, dtype: str):
    from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
    from diffsynth.core.loader.config import ModelConfig
    from diffsynth.utils.lora.general import GeneralLoRALoader

    device, dt = _device_dtype(device, dtype)
    base = "HiDream-ai/HiDream-O1-Image-Dev"
    pipe = HiDreamO1ImagePipeline.from_pretrained(
        torch_dtype=dt, device=device,
        model_configs=[ModelConfig(model_id=base, origin_file_pattern="model-*.safetensors")],
        processor_config=ModelConfig(model_id=base, origin_file_pattern="./"),
    )
    return pipe, GeneralLoRALoader(device="cpu")


def build_flux2(device: str, dtype: str):
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    from diffsynth.utils.lora.general import GeneralLoRALoader

    device, dt = _device_dtype(device, dtype)
    base = "black-forest-labs/FLUX.2-dev"
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=dt, device=device,
        model_configs=[
            ModelConfig(model_id=base, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=base, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=base, origin_file_pattern="tokenizer/"),
    )
    return pipe, GeneralLoRALoader(device="cpu")


BUILDERS: Dict[str, Callable] = {
    "flux": build_flux,
    "qwen": build_qwen,
    "z_image": build_z_image,
    "hidream": build_hidream,
    "flux2": build_flux2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(BUILDERS), default="flux")
    parser.add_argument("--loras", nargs="+", default=None, help="LoRA paths or HF org/repo[:filename] specs.")
    parser.add_argument("--prompts", nargs="+", default=None, help="One calibration prompt per LoRA.")
    parser.add_argument("--output", default=None, help="Output safetensors path (default: merged_<backbone>.safetensors).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(_DTYPE), default="bf16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    loras = args.loras or DEFAULT_LORAS
    prompts = args.prompts or DEFAULT_PROMPTS
    if len(loras) != len(prompts):
        parser.error(f"--loras ({len(loras)}) and --prompts ({len(prompts)}) must match.")
    if args.loras is None:
        if args.backbone != "flux":
            parser.error(
                "The default cat/dog LoRAs are FLUX-only. "
                f"Pass --loras and --prompts when using --backbone {args.backbone}."
            )
        ensure_demo_loras()

    output = os.path.abspath(args.output or f"merged_{args.backbone}.safetensors")

    logging.info("Building %s pipeline...", args.backbone)
    pipe, loader = BUILDERS[args.backbone](args.device, args.dtype)

    run(pipe=pipe, loader=loader, loras=loras, prompts=prompts, output=output)
    print(f"\nMerged LoRA written to: {output}")


if __name__ == "__main__":
    main()
