"""
Build a flat ComfyUI API prompt for the Wan2.2 Animate Character Replace workflow.

This manually flattens the subgraph-based UI workflow into the API prompt format.
"""


def build_prompt(
    video: str = "input.mp4",
    image: str = "input.png",
    width: int = 832,
    height: int = 480,
    prompt: str = "the person is dancing",
    length: int = 77,
    seed: int = -1,
    steps: int = 6,
    cfg: float = 1.0,
):
    """
    Build the API prompt.

    Args:
        video: Filename of uploaded input video
        image: Filename of uploaded reference character image
        width: Output width
        height: Output height
        prompt: Positive prompt describing the action
        length: Number of frames to generate
        seed: Random seed (-1 for random)
        steps: Sampling steps
        cfg: CFG scale
    """
    import random
    if seed == -1:
        seed = random.randint(0, 2**53)

    return {
        # --- Top-level nodes (outside subgraphs) ---
        # VHS_LoadVideo - Input Video
        "301": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video,
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        # VHS_VideoInfo
        "314": {
            "class_type": "VHS_VideoInfo",
            "inputs": {
                "video_info": ["301", 3],
            },
        },
        # LoadImage - Reference Image
        "10": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image,
            },
        },

        # --- Main subgraph nodes (prefixed 1xxx to avoid collisions) ---
        # UNETLoader
        "1020": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                "weight_dtype": "default",
            },
        },
        # CLIPLoader
        "1002": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        # VAELoader
        "1003": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors",
            },
        },
        # CLIPVisionLoader
        "1004": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "clip_vision_h.safetensors",
            },
        },
        # LoraLoaderModelOnly - lightx2v
        "1018": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
                "strength_model": 1.0,
                "model": ["1020", 0],
            },
        },
        # LoraLoaderModelOnly - relight
        "1099": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "WanAnimate_relight_lora_fp16.safetensors",
                "strength_model": 1.0,
                "model": ["1018", 0],
            },
        },
        # ModelSamplingSD3
        "1060": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "shift": 8,
                "model": ["1099", 0],
            },
        },
        # CLIPTextEncode - Negative
        "1001": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，",
                "clip": ["1002", 0],
            },
        },
        # CLIPTextEncode - Positive
        "1021": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1002", 0],
            },
        },
        # ImageScale - Resize Input Video
        "1338": {
            "class_type": "ImageScale",
            "inputs": {
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
                "image": ["301", 0],
            },
        },
        # ImageScale - Resize Reference Image
        "1342": {
            "class_type": "ImageScale",
            "inputs": {
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
                "image": ["10", 0],
            },
        },
        # CLIPVisionEncode
        "1009": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "crop": "none",
                "clip_vision": ["1004", 0],
                "image": ["1342", 0],
            },
        },
        # GetImageSize
        "1348": {
            "class_type": "GetImageSize",
            "inputs": {
                "image": ["1338", 0],
            },
        },
        # OnnxDetectionModelLoader
        "1328": {
            "class_type": "OnnxDetectionModelLoader",
            "inputs": {
                "vitpose_model": "vitpose-l-wholebody.onnx",
                "yolo_model": "yolov10m.onnx",
                "onnx_device": "CUDAExecutionProvider",
            },
        },
        # PoseAndFaceDetection
        "1334": {
            "class_type": "PoseAndFaceDetection",
            "inputs": {
                "width": ["1348", 0],
                "height": ["1348", 1],
                "model": ["1328", 0],
                "images": ["1338", 0],
            },
        },
        # DownloadAndLoadSAM2Model
        "1324": {
            "class_type": "DownloadAndLoadSAM2Model",
            "inputs": {
                "model": "sam2.1_hiera_base_plus.safetensors",
                "segmentor": "video",
                "device": "cuda",
                "precision": "fp16",
            },
        },
        # Sam2Segmentation
        "1326": {
            "class_type": "Sam2Segmentation",
            "inputs": {
                "keep_model_loaded": False,
                "sam2_model": ["1324", 0],
                "image": ["1338", 0],
                "bboxes": ["1334", 3],
            },
        },
        # GrowMaskWithBlur
        "1327": {
            "class_type": "GrowMaskWithBlur",
            "inputs": {
                "expand": 10,
                "incremental_expandrate": 0,
                "tapered_corners": True,
                "flip_input": False,
                "blur_radius": 0,
                "lerp_alpha": 1,
                "decay_factor": 1,
                "fill_holes": False,
                "mask": ["1326", 0],
            },
        },
        # BlockifyMask
        "1325": {
            "class_type": "BlockifyMask",
            "inputs": {
                "block_size": 32,
                "device": "cpu",
                "masks": ["1327", 0],
            },
        },
        # DrawMaskOnImage
        "1332": {
            "class_type": "DrawMaskOnImage",
            "inputs": {
                "color": "0, 0, 0",
                "device": "cpu",
                "image": ["1338", 0],
                "mask": ["1325", 0],
            },
        },
        # DrawViTPose
        "1335": {
            "class_type": "DrawViTPose",
            "inputs": {
                "width": ["1348", 0],
                "height": ["1348", 1],
                "retarget_padding": 16,
                "body_stick_width": -1,
                "hand_stick_width": -1,
                "draw_head": True,
                "pose_data": ["1334", 0],
            },
        },
        # CreateVideo - pose video
        "1340": {
            "class_type": "CreateVideo",
            "inputs": {
                "fps": ["314", 0],
                "images": ["1335", 0],
            },
        },
        # CreateVideo - face detection video
        "1333": {
            "class_type": "CreateVideo",
            "inputs": {
                "fps": ["314", 0],
                "images": ["1334", 1],
            },
        },
        # CreateVideo - mask overlay video
        "1339": {
            "class_type": "CreateVideo",
            "inputs": {
                "fps": ["314", 0],
                "images": ["1332", 0],
            },
        },
        # SaveVideo - pose
        "1330": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": "video/ComfyUI",
                "format": "auto",
                "codec": "auto",
                "video": ["1340", 0],
            },
        },
        # SaveVideo - face
        "1329": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": "video/ComfyUI",
                "format": "auto",
                "codec": "auto",
                "video": ["1333", 0],
            },
        },
        # SaveVideo - mask
        "1331": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": "video/ComfyUI",
                "format": "auto",
                "codec": "auto",
                "video": ["1339", 0],
            },
        },

        # --- Inner subgraph nodes (prefixed 2xxx) ---
        # WanAnimateToVideo
        "2062": {
            "class_type": "WanAnimateToVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "continue_motion_max_frames": 5,
                "video_frame_offset": 0,
                "positive": ["1021", 0],
                "negative": ["1001", 0],
                "vae": ["1003", 0],
                "clip_vision_output": ["1009", 0],
                "reference_image": ["1342", 0],
                "face_video": ["1334", 1],
                "pose_video": ["1335", 0],
                "background_video": ["1332", 0],
                "character_mask": ["1325", 0],
            },
        },
        # KSampler
        "2063": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "control_after_generate": "randomize",
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["1060", 0],
                "positive": ["2062", 0],
                "negative": ["2062", 1],
                "latent_image": ["2062", 2],
            },
        },
        # TrimVideoLatent
        "2057": {
            "class_type": "TrimVideoLatent",
            "inputs": {
                "trim_index": 0,
                "latent": ["2063", 0],
                "trim_count": ["2062", 3],
            },
        },
        # VAEDecode
        "2058": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["2057", 0],
                "vae": ["1003", 0],
            },
        },
        # ImageFromBatch
        "2230": {
            "class_type": "ImageFromBatch",
            "inputs": {
                "batch_index": 0,
                "length": 4096,
                "image": ["2058", 0],
            },
        },
        # CreateVideo - final output
        "2015": {
            "class_type": "CreateVideo",
            "inputs": {
                "fps": ["314", 0],
                "images": ["2230", 0],
            },
        },
        # SaveVideo - final output (this is the main output)
        "19": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": "video/ComfyUI",
                "format": "auto",
                "codec": "auto",
                "video": ["2015", 0],
            },
        },
    }
