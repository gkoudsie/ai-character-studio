"""
ComfyUI HTTP/WebSocket client.

Talks to a locally running ComfyUI instance. This is the only place that knows
how to communicate with ComfyUI, so the rest of the pipeline stays clean.

Flow:
  1. Upload any input images (reference / source frames) via /upload/image.
  2. Submit a workflow graph (a dict) via /prompt, which returns a prompt_id.
  3. Listen on the websocket for an "executing" message with node == None and
     the matching prompt_id, which means the run finished.
  4. Fetch /history/{prompt_id} to find output files, then download them
     via /view.

ComfyUI is free, open source, and cross-platform, so nothing here is tied to
any particular OS or cloud provider.
"""

from __future__ import annotations

import io
import json
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import requests
import websocket  # from websocket-client


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI cannot be reached or a run fails."""


class ComfyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        timeout: int = 600,
        base_url: str | None = None,
    ):
        """
        Connect to ComfyUI.

        Either pass host+port (the local case), or pass a full base_url such as
        "https://abc123.trycloudflare.com" for a remote/cloud ComfyUI. When a
        base_url is given, the websocket URL is derived from it (http->ws,
        https->wss) so both plain and TLS endpoints work.
        """
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

        if base_url:
            base_url = base_url.rstrip("/")
            parsed = urllib.parse.urlparse(base_url)
            scheme = parsed.scheme or "http"
            self.host = parsed.hostname or host
            self.port = parsed.port or (443 if scheme == "https" else 80)
            self.base_http = base_url
            ws_scheme = "wss" if scheme == "https" else "ws"
            self.base_ws = f"{ws_scheme}://{parsed.netloc}/ws"
        else:
            self.host = host
            self.port = port
            self.base_http = f"http://{host}:{port}"
            self.base_ws = f"ws://{host}:{port}/ws"

    # ------------------------------------------------------------------ #
    # Connectivity
    # ------------------------------------------------------------------ #
    def check_connection(self) -> None:
        """Raise a clear error early if ComfyUI is not running."""
        try:
            resp = requests.get(f"{self.base_http}/system_stats", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIError(
                f"Could not reach ComfyUI at {self.base_http}. "
                f"Is ComfyUI running? Start it, then try again.\n  Details: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Uploading input images
    # ------------------------------------------------------------------ #
    def upload_image(self, image_path: str | Path, subfolder: str = "") -> str:
        """
        Upload a local image into ComfyUI's input folder.

        Returns the filename (as ComfyUI sees it) to reference inside a workflow.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise ComfyUIError(f"Input image not found: {image_path}")

        with image_path.open("rb") as fh:
            files = {"image": (image_path.name, fh, "application/octet-stream")}
            data = {"overwrite": "true"}
            if subfolder:
                data["subfolder"] = subfolder
            try:
                resp = requests.post(
                    f"{self.base_http}/upload/image",
                    files=files,
                    data=data,
                    timeout=60,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise ComfyUIError(f"Failed to upload image {image_path}: {exc}") from exc

        info = resp.json()
        name = info.get("name", image_path.name)
        sub = info.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    # ------------------------------------------------------------------ #
    # Submitting a workflow and waiting for it to finish
    # ------------------------------------------------------------------ #
    def submit(self, workflow: dict[str, Any]) -> str:
        """Queue a workflow. Returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            resp = requests.post(f"{self.base_http}/prompt", json=payload, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            # ComfyUI returns a helpful JSON error body on validation failures.
            detail = ""
            if exc.response is not None:
                try:
                    detail = json.dumps(exc.response.json(), indent=2)
                except ValueError:
                    detail = exc.response.text
            raise ComfyUIError(f"ComfyUI rejected the workflow: {exc}\n{detail}") from exc

        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            raise ComfyUIError("ComfyUI did not return a prompt_id.")
        return prompt_id

    def _await_completion(self, prompt_id: str) -> None:
        """Block until the given prompt_id finishes executing."""
        ws = websocket.WebSocket()
        ws_url = f"{self.base_ws}?clientId={self.client_id}"
        try:
            ws.connect(ws_url, timeout=30)
        except Exception as exc:  # noqa: BLE001 - websocket raises many types
            raise ComfyUIError(f"Could not open ComfyUI websocket: {exc}") from exc

        deadline = time.time() + self.timeout
        try:
            while time.time() < deadline:
                ws.settimeout(max(1.0, deadline - time.time()))
                try:
                    message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not isinstance(message, str):
                    # Binary previews; ignore.
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "executing":
                    body = data.get("data", {})
                    if body.get("prompt_id") == prompt_id and body.get("node") is None:
                        return  # node == None => this prompt is done
                elif data.get("type") == "execution_error":
                    body = data.get("data", {})
                    if body.get("prompt_id") == prompt_id:
                        raise ComfyUIError(
                            f"ComfyUI reported an execution error:\n"
                            f"{json.dumps(body, indent=2)}"
                        )
            raise ComfyUIError(
                f"Timed out after {self.timeout}s waiting for ComfyUI to finish "
                f"(prompt_id={prompt_id})."
            )
        finally:
            ws.close()

    # ------------------------------------------------------------------ #
    # Collecting outputs
    # ------------------------------------------------------------------ #
    def _history(self, prompt_id: str) -> dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_http}/history/{prompt_id}", timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIError(f"Failed to fetch history for {prompt_id}: {exc}") from exc
        return resp.json().get(prompt_id, {})

    def _download_output(self, meta: dict[str, str]) -> bytes:
        params = urllib.parse.urlencode(
            {
                "filename": meta.get("filename", ""),
                "subfolder": meta.get("subfolder", ""),
                "type": meta.get("type", "output"),
            }
        )
        try:
            resp = requests.get(f"{self.base_http}/view?{params}", timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIError(f"Failed to download output {meta}: {exc}") from exc
        return resp.content

    def run(self, workflow: dict[str, Any]) -> list[tuple[str, bytes]]:
        """
        Submit a workflow, wait for it, and return all produced files.

        Returns a list of (filename, bytes). Covers both images and videos/gifs,
        since ComfyUI records them under different history keys depending on the
        output node used.
        """
        prompt_id = self.submit(workflow)
        self._await_completion(prompt_id)
        history = self._history(prompt_id)
        outputs = history.get("outputs", {})

        collected: list[tuple[str, bytes]] = []
        for node_output in outputs.values():
            # Image-style outputs.
            for key in ("images", "gifs", "videos"):
                for item in node_output.get(key, []) or []:
                    # Skip pure preview/temp artifacts we don't want to save.
                    if item.get("type") == "temp":
                        continue
                    filename = item.get("filename", "")
                    if not filename:
                        continue
                    collected.append((filename, self._download_output(item)))

        if not collected:
            raise ComfyUIError(
                f"Run finished but produced no downloadable outputs "
                f"(prompt_id={prompt_id}). Check that the workflow has a "
                f"SaveImage / video output node."
            )
        return collected


def save_bytes(data: bytes, dest: str | Path) -> Path:
    """Small helper to write downloaded bytes to disk, creating dirs as needed."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        fh.write(data)
    return dest


# Keeps the io import meaningful for type-checkers / future streaming use.
_ = io
