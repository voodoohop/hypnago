APP_NAME = "airt-machine"
GPU = "H200"
TIMEOUT = 3600

MODELS = [
    # Z-Image Turbo
    (
        "diffusion_models",
        "z_image_turbo_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    ),
    (
        "text_encoders",
        "qwen_3_4b.safetensors",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
    ),
    (
        "vae",
        "ae.safetensors",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    ),
    # Wan 2.2 I2V
    (
        "diffusion_models",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    ),
    (
        "diffusion_models",
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
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
    # LightX2V LoRAs for 4-step acceleration
    (
        "loras",
        "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "https://huggingface.co/BANODOCO/wan2.2-i2v-lightx2v-4step-loras/resolve/main/rank64_original/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
    ),
    (
        "loras",
        "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "https://huggingface.co/BANODOCO/wan2.2-i2v-lightx2v-4step-loras/resolve/main/rank64_original/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
    ),
    # Upscaler
    (
        "upscale_models",
        "4x-UltraSharp.safetensors",
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors",
    ),
]

# Custom nodes required by this workflow (org/repo for git clone)
CUSTOM_NODES = [
    "1038lab/ComfyUI-QwenVL",
    "ltdrdata/ComfyUI-Impact-Pack",
    "kijai/ComfyUI-KJNodes",
    "Kosinkadink/ComfyUI-VideoHelperSuite",
]
