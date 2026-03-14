APP_NAME = "wan22-character-replace"
GPU = "H200"
TIMEOUT = 3600

MODELS = [
    (
        "loras",
        "WanAnimate_relight_lora_fp16.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors",
    ),
    (
        "diffusion_models",
        "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
    ),
    (
        "text_encoders",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    (
        "vae",
        "wan_2.1_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    ),
    (
        "clip_vision",
        "clip_vision_h.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
    ),
    (
        "loras",
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    ),
]

# Custom nodes required by this workflow (org/repo for git clone)
CUSTOM_NODES = [
    "Kosinkadink/ComfyUI-VideoHelperSuite",
    "kijai/ComfyUI-segment-anything-2",
    "kijai/ComfyUI-WanAnimatePreprocess",
    "kijai/ComfyUI-KJNodes",
]
