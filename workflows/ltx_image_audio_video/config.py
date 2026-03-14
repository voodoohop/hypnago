APP_NAME = "ltx23-image-audio-video"
GPU = "A100-80GB"
TIMEOUT = 3600

MODELS = [
    (
        "checkpoints",
        "ltx-2.3-22b-dev-fp8.safetensors",
        "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors",
    ),
    (
        "text_encoders",
        "gemma_3_12B_it_fp4_mixed.safetensors",
        "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    ),
    (
        "loras",
        "ltx-2.3-22b-distilled-lora-384.safetensors",
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors",
    ),
    (
        "loras",
        "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
    ),
    (
        "latent_upscale_models",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    ),
]

# Custom nodes required by this workflow (org/repo for git clone)
CUSTOM_NODES = [
    "Lightricks/ComfyUI-LTXVideo",
]
