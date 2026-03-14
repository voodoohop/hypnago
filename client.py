"""
Client for ComfyUI workflows deployed on Modal.

Usage:
    python client.py --workflow wan22_character_replace \
        --video input.mp4 \
        --image character.jpg

    python client.py --workflow ltx_image_audio_video \
        --image input.jpg \
        --audio speech.mp3

    python client.py --workflow airt_machine \
        --image input.jpg
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

# Base URLs for each workflow
ENDPOINTS = {
    "wan22_character_replace": "https://pollinations--wan22-character-replace-ui.modal.run",
    "ltx_image_audio_video": "https://pollinations--ltx23-image-audio-video-ui.modal.run",
    "airt_machine": "https://pollinations--airt-machine-ui.modal.run",
}


def upload_file(base_url, filepath, subfolder="", file_type="input"):
    """Upload a file to ComfyUI's /upload/image or /upload/video endpoint."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".mp4", ".webm", ".mov", ".avi"):
        url = f"{base_url}/upload/image"  # ComfyUI uses same endpoint
    else:
        url = f"{base_url}/upload/image"

    # Build multipart form data
    boundary = uuid.uuid4().hex
    data = bytearray()

    # File field
    data += f"--{boundary}\r\n".encode()
    data += f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
    content_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".mp4": "video/mp4", ".webm": "video/webm", ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
    }.get(ext, "application/octet-stream")
    data += f"Content-Type: {content_type}\r\n\r\n".encode()
    with open(filepath, "rb") as f:
        data += f.read()
    data += b"\r\n"

    # Subfolder field
    if subfolder:
        data += f"--{boundary}\r\n".encode()
        data += f'Content-Disposition: form-data; name="subfolder"\r\n\r\n{subfolder}\r\n'.encode()

    # Type field
    data += f"--{boundary}\r\n".encode()
    data += f'Content-Disposition: form-data; name="type"\r\n\r\n{file_type}\r\n'.encode()

    # Overwrite field
    data += f"--{boundary}\r\n".encode()
    data += f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode()

    data += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=bytes(data),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    print(f"  Uploaded {filename}: {result}")
    return result


def queue_prompt(base_url, prompt, client_id=None):
    """Queue a prompt for execution."""
    if client_id is None:
        client_id = str(uuid.uuid4())

    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    return result, client_id


def poll_status(base_url, prompt_id, timeout=600):
    """Poll until the prompt is done."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{base_url}/api/history/{prompt_id}", timeout=30)
            history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except Exception:
            pass
        time.sleep(5)
        elapsed = int(time.time() - start)
        print(f"  Waiting... ({elapsed}s)", end="\r")
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def download_output(base_url, output_info, output_dir="."):
    """Download output files (videos/images) from the history."""
    outputs = output_info.get("outputs", {})
    downloaded = []
    for node_id, node_output in outputs.items():
        for file_type in ["videos", "images", "gifs"]:
            for item in node_output.get(file_type, []):
                filename = item["filename"]
                subfolder = item.get("subfolder", "")
                params = urllib.parse.urlencode({
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": item.get("type", "output"),
                })
                url = f"{base_url}/api/view?{params}"
                out_path = os.path.join(output_dir, filename)
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, out_path)
                downloaded.append(out_path)
                print(f"  Saved to {out_path}")
    return downloaded


def build_wan22_prompt(video_filename, image_filename, width=720, height=720, length=81):
    """Build the API prompt for Wan2.2 Character Replace."""
    return {
        # VHS_LoadVideo - Input Video (node 301)
        "301": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_filename,
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        # LoadImage - Reference Image (node 10)
        "10": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_filename,
            },
        },
        # PrimitiveInt - Width (node 159)
        "159": {
            "class_type": "PrimitiveInt",
            "inputs": {
                "value": width,
            },
        },
        # PrimitiveInt - Height (node 160)
        "160": {
            "class_type": "PrimitiveInt",
            "inputs": {
                "value": height,
            },
        },
        # SaveVideo (node 19)
        "19": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": "video/ComfyUI",
                "format": "auto",
                "codec": "auto",
            },
        },
        # VHS_VideoInfo (node 314) - gets info from loaded video
        "314": {
            "class_type": "VHS_VideoInfo",
            "inputs": {
                "video_info": ["301", 1],
            },
        },
        # Main subgraph node (node 344) - the embedded workflow component
        "344": {
            "class_type": "b396d3e6-70cc-4a91-81db-6e0399a4edb6",
            "inputs": {
                "fps": ["314", 0],
                "image": ["301", 0],
                "width": ["159", 0],
                "height": ["160", 0],
                "image_1": ["10", 0],
                "length": length,
            },
        },
    }


def run_wan22(args):
    base_url = ENDPOINTS["wan22_character_replace"]
    print("Uploading files...")
    upload_file(base_url, args.video)
    upload_file(base_url, args.image)

    video_name = os.path.basename(args.video)
    image_name = os.path.basename(args.image)

    print("Building prompt...")
    prompt = build_wan22_prompt(
        video_filename=video_name,
        image_filename=image_name,
        width=args.width,
        height=args.height,
    )

    print("Queuing prompt...")
    result, client_id = queue_prompt(base_url, prompt)
    prompt_id = result["prompt_id"]
    print(f"  Prompt ID: {prompt_id}")

    print("Waiting for completion...")
    output = poll_status(base_url, prompt_id, timeout=args.timeout)

    print("Downloading output...")
    files = download_output(base_url, output, args.output_dir)
    print(f"Done! Output files: {files}")


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Modal Client")
    parser.add_argument("--workflow", required=True, choices=list(ENDPOINTS.keys()))
    parser.add_argument("--video", help="Input video file")
    parser.add_argument("--image", help="Input image file")
    parser.add_argument("--audio", help="Input audio file")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    if args.workflow == "wan22_character_replace":
        if not args.video or not args.image:
            parser.error("wan22_character_replace requires --video and --image")
        run_wan22(args)
    else:
        print(f"Client for {args.workflow} not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
