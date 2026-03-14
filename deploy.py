"""
Generic ComfyUI workflow deployer for Modal.

Usage:
    # First time: download models to volume (one-time, ~10-30min)
    WORKFLOW=wan22_character_replace modal run deploy.py::setup

    # Deploy (fast, ~10s since models are on volume)
    WORKFLOW=wan22_character_replace modal deploy deploy.py

    # Dev mode (hot reload on code changes)
    WORKFLOW=wan22_character_replace modal serve deploy.py
"""

import os
import subprocess

import modal

WORKFLOW = os.environ.get("WORKFLOW", "wan22_character_replace")
COMFYUI_DIR = "/root/comfy/ComfyUI"

# --- Load config locally ---
_here = os.path.dirname(os.path.abspath(__file__))
_config_path = os.path.join(_here, "workflows", WORKFLOW, "config.py")

if os.path.exists(_config_path):
    import importlib, sys
    sys.path.insert(0, _here)
    config = importlib.import_module(f"workflows.{WORKFLOW}.config")
    _APP_NAME = config.APP_NAME
    _GPU = config.GPU
    _TIMEOUT = getattr(config, "TIMEOUT", 3600)
    _MODELS = config.MODELS
    _CUSTOM_NODES = getattr(config, "CUSTOM_NODES", [])
else:
    _APP_NAME = os.environ.get("MODAL_APP_NAME", "comfyui-workflow")
    _GPU = os.environ.get("MODAL_GPU", "H200")
    _TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "3600"))
    _MODELS = []
    _CUSTOM_NODES = []

# --- Persistent volume for models ---
models_vol = modal.Volume.from_name(f"{_APP_NAME}-models", create_if_missing=True)

# --- Image: ComfyUI + custom nodes + wrapper deps ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg")
    .pip_install("comfy-cli==1.6.0", "fastapi[standard]", "httpx")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
)

# Install custom nodes
node_install_commands = []
for node in _CUSTOM_NODES:
    node_name = node.split("/")[-1]
    node_dir = f"{COMFYUI_DIR}/custom_nodes/{node_name}"
    if "/" in node:
        repo_url = f"https://github.com/{node}.git"
        node_install_commands.append(
            f"git clone {repo_url} {node_dir} && "
            f"(cd {node_dir} && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi)"
        )
    else:
        node_install_commands.append(f"comfy --skip-prompt node registry-install {node}")

if node_install_commands:
    image = image.run_commands(*node_install_commands)

# Workflow JSON last
workflow_json = os.path.join(_here, "workflows", WORKFLOW, "workflow.json")
if os.path.exists(workflow_json):
    image = image.add_local_file(workflow_json, f"{COMFYUI_DIR}/user/default/workflows/workflow.json")

# Also add the workflow-specific API prompt builder if it exists
api_prompt_path = os.path.join(_here, "workflows", WORKFLOW, "api_prompt.py")
if os.path.exists(api_prompt_path):
    image = image.add_local_file(api_prompt_path, "/root/api_prompt.py")

app = modal.App(name=_APP_NAME, image=image)


# --- One-time setup: download models to volume ---
@app.function(
    volumes={"/vol/models": models_vol},
    timeout=3600,
)
def setup():
    """Run once to download models into the persistent volume."""
    import concurrent.futures

    def download_model(subfolder, filename, url):
        dest_dir = f"/vol/models/{subfolder}"
        dest = f"{dest_dir}/{filename}"
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  Already exists: {filename}")
            return
        os.makedirs(dest_dir, exist_ok=True)
        print(f"  Downloading {filename}...")
        subprocess.run(["wget", "-q", "-O", dest, url], check=True)
        print(f"  Done: {filename}")

    print(f"Downloading {len(_MODELS)} models...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download_model, *m) for m in _MODELS]
        concurrent.futures.wait(futures)
        for f in futures:
            f.result()

    models_vol.commit()
    print("Models committed to volume.")
    print("Setup complete! Now deploy with: modal deploy deploy.py")


# --- Main server: ComfyUI + simple API wrapper ---
@app.function(
    max_containers=1,
    gpu=_GPU,
    timeout=_TIMEOUT,
    container_idle_timeout=300,
    volumes={"/vol/models": models_vol},
)
@modal.asgi_app()
def api():
    import asyncio
    import base64
    import json
    import time
    import uuid

    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    COMFY = "http://127.0.0.1:8000"
    web = FastAPI()

    # --- Start ComfyUI in the background ---
    # Symlink models from volume
    models_dir = f"{COMFYUI_DIR}/models"
    for subfolder in os.listdir("/vol/models"):
        src = f"/vol/models/{subfolder}"
        dst = f"{models_dir}/{subfolder}"
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for fname in os.listdir(src):
                src_file = f"{src}/{fname}"
                dst_file = f"{dst}/{fname}"
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)

    subprocess.Popen(
        "comfy launch -- --listen 127.0.0.1 --port 8000",
        shell=True,
    )

    # --- Load API prompt builder if available ---
    build_prompt_fn = None
    if os.path.exists("/root/api_prompt.py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("api_prompt", "/root/api_prompt.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build_prompt_fn = mod.build_prompt

    async def wait_for_comfy(timeout=120):
        """Wait for ComfyUI to be ready."""
        async with httpx.AsyncClient() as client:
            for _ in range(timeout):
                try:
                    r = await client.get(f"{COMFY}/api/system_stats", timeout=2)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(1)
        return False

    async def download_url(url: str) -> tuple[bytes, str]:
        """Download a file from a URL, return (bytes, filename)."""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(url, timeout=120)
            r.raise_for_status()
            # Extract filename from URL or content-disposition
            fname = url.split("/")[-1].split("?")[0]
            if not fname or "." not in fname:
                ct = r.headers.get("content-type", "")
                ext = {"image/jpeg": ".jpg", "image/png": ".png", "video/mp4": ".mp4",
                       "audio/mpeg": ".mp3", "audio/wav": ".wav"}.get(ct, ".bin")
                fname = f"input_{uuid.uuid4().hex[:8]}{ext}"
            return r.content, fname

    async def upload_to_comfy(data: bytes, filename: str):
        """Upload a file to ComfyUI."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{COMFY}/upload/image",
                files={"image": (filename, data)},
                data={"overwrite": "true"},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()

    async def queue_prompt(prompt: dict) -> str:
        """Queue a prompt, return prompt_id."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{COMFY}/api/prompt",
                json={"prompt": prompt},
                timeout=60,
            )
            if r.status_code != 200:
                raise Exception(f"ComfyUI rejected prompt ({r.status_code}): {r.text}")
            return r.json()["prompt_id"]

    async def poll_result(prompt_id: str, timeout: int = 600) -> dict:
        """Poll until the prompt completes."""
        async with httpx.AsyncClient() as client:
            deadline = time.time() + timeout
            while time.time() < deadline:
                r = await client.get(f"{COMFY}/api/history/{prompt_id}", timeout=10)
                if r.status_code == 200:
                    history = r.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("completed") or status.get("status_str") == "success":
                            return history[prompt_id]
                        if status.get("status_str") == "error":
                            return history[prompt_id]
                await asyncio.sleep(2)
        raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")

    async def get_output_files(result: dict) -> list[dict]:
        """Extract output file info from result."""
        files = []
        for node_id, node_output in result.get("outputs", {}).items():
            for ftype in ["videos", "images", "gifs"]:
                for item in node_output.get(ftype, []):
                    files.append({
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                        "format": ftype,
                    })
        return files

    # ---- API Endpoints ----

    @web.get("/health")
    async def health():
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{COMFY}/api/system_stats", timeout=5)
                return r.json()
        except Exception as e:
            return JSONResponse({"status": "starting", "error": str(e)}, status_code=503)

    @web.post("/run")
    async def run(request: Request):
        """
        Simple API: pass params, get back output.

        POST /run
        {
            "image_url": "https://...",      // reference image URL
            "video_url": "https://...",      // input video URL (if needed)
            "audio_url": "https://...",      // input audio URL (if needed)
            "width": 720,
            "height": 720,
            ...any other workflow params...
        }

        Returns:
        {
            "status": "success",
            "outputs": [{"url": "/output/...", "filename": "..."}]
        }
        """
        try:
            if not await wait_for_comfy(timeout=180):
                return JSONResponse({"error": "ComfyUI not ready"}, status_code=503)

            body = await request.json()

            # Download and upload input files
            filenames = {}
            for key in ["image_url", "video_url", "audio_url"]:
                url = body.get(key)
                if url:
                    data, fname = await download_url(url)
                    await upload_to_comfy(data, fname)
                    filenames[key.replace("_url", "")] = fname

            # Also support base64 inputs
            for key in ["image_base64", "video_base64", "audio_base64"]:
                b64 = body.get(key)
                if b64:
                    media_type = key.replace("_base64", "")
                    ext = {"image": ".png", "video": ".mp4", "audio": ".mp3"}[media_type]
                    fname = f"input_{uuid.uuid4().hex[:8]}{ext}"
                    data = base64.b64decode(b64)
                    await upload_to_comfy(data, fname)
                    filenames[media_type] = fname

            # Build the prompt
            if build_prompt_fn is None:
                return JSONResponse(
                    {"error": "No api_prompt.py found for this workflow. Use /comfy/* to access ComfyUI directly."},
                    status_code=400,
                )

            params = {k: v for k, v in body.items() if not k.endswith("_url") and not k.endswith("_base64")}
            params.update(filenames)
            prompt = build_prompt_fn(**params)

            # Queue and wait
            prompt_id = await queue_prompt(prompt)
            result = await poll_result(prompt_id, timeout=body.get("timeout", 600))

            status = result.get("status", {})
            if status.get("status_str") == "error":
                return JSONResponse({"error": "Workflow execution failed", "details": result}, status_code=500)

            # Get output files
            output_files = await get_output_files(result)

            # Return output URLs (relative to this server)
            outputs = []
            for f in output_files:
                outputs.append({
                    "url": f"/output/{f['subfolder']}/{f['filename']}" if f['subfolder'] else f"/output/{f['filename']}",
                    "filename": f["filename"],
                    "format": f["format"],
                })

            return {"status": "success", "prompt_id": prompt_id, "outputs": outputs}

        except Exception as e:
            import traceback
            return JSONResponse(
                {"error": str(e), "traceback": traceback.format_exc()},
                status_code=500,
            )

    @web.get("/output/{path:path}")
    async def get_output(path: str):
        """Serve output files from ComfyUI."""
        async with httpx.AsyncClient() as client:
            # Parse path into filename and subfolder
            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                subfolder, filename = parts
            else:
                subfolder, filename = "", parts[0]
            params = {"filename": filename, "subfolder": subfolder, "type": "output"}
            r = await client.get(f"{COMFY}/api/view", params=params, timeout=60)
            r.raise_for_status()
            ct = r.headers.get("content-type", "application/octet-stream")
            return Response(content=r.content, media_type=ct)

    # --- Proxy everything else to ComfyUI (so the UI still works) ---
    @web.api_route("/comfy/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def comfy_proxy(request: Request, path: str):
        """Proxy to ComfyUI for direct access."""
        async with httpx.AsyncClient() as client:
            url = f"{COMFY}/{path}"
            r = await client.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=await request.body(),
                params=dict(request.query_params),
                timeout=120,
            )
            return Response(content=r.content, status_code=r.status_code,
                          headers={k: v for k, v in r.headers.items() if k.lower() not in ("transfer-encoding",)})

    return web
