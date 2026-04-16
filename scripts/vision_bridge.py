#!/usr/bin/env python3
"""
Vision bridge for OpenCode: when the UI/agent path is text-only, call the local
OpenAI-compatible vLLM with proper multimodal content (text + image_url data URIs).

Default endpoint matches opencode Nebius config: http://127.0.0.1:6767/v1

Examples:
  .venv/bin/python scripts/vision_bridge.py "Describe this figure" --image figures/plot.png
  .venv/bin/python scripts/vision_bridge.py "Is the title readable?" --pdf report.pdf --page 0
  .venv/bin/python scripts/vision_bridge.py "Compare" -i a.png -i b.png
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE = "http://127.0.0.1:6767/v1"
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_API_KEY = "dummy"


def _data_url_from_image_file(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _data_url_from_pdf_page(pdf_path: Path, page_0based: int, dpi: int) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise SystemExit(
            "PDF pages need PyMuPDF: uv pip install --python .venv/bin/python pymupdf"
        ) from e
    doc = fitz.open(pdf_path)
    if page_0based < 0 or page_0based >= len(doc):
        doc.close()
        raise SystemExit(f"page {page_0based} out of range (0..{len(doc) - 1})")
    page = doc.load_page(page_0based)
    pix = page.get_pixmap(dpi=dpi)
    png = pix.tobytes("png")
    doc.close()
    b64 = base64.standard_b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _post_chat(
    base_url: str,
    api_key: str,
    model: str,
    user_content: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {err}") from e
    except URLError as e:
        raise SystemExit(f"Request failed: {e}") from e

    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(payload, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Multimodal chat via local OpenAI-compatible vLLM.")
    p.add_argument("prompt", help="Question / instruction for the model")
    p.add_argument(
        "-i",
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="Image file (repeat for multiple)",
    )
    p.add_argument("--pdf", type=Path, help="PDF file; use with --page")
    p.add_argument(
        "--page",
        type=int,
        default=0,
        help="0-based page index when using --pdf (default: 0)",
    )
    p.add_argument("--dpi", type=int, default=150, help="Rasterize PDF at this DPI (default: 150)")
    p.add_argument("--base-url", default=DEFAULT_BASE, help=f"API base (default: {DEFAULT_BASE})")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    p.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer token (default: dummy)")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    content: list[dict] = [{"type": "text", "text": args.prompt}]

    for img in args.image:
        path = Path(img).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
        url = _data_url_from_image_file(path)
        content.append({"type": "image_url", "image_url": {"url": url}})

    if args.pdf is not None:
        pdf_path = args.pdf.expanduser().resolve()
        if not pdf_path.is_file():
            raise SystemExit(f"Not a file: {pdf_path}")
        url = _data_url_from_pdf_page(pdf_path, args.page, args.dpi)
        content.append({"type": "image_url", "image_url": {"url": url}})

    if len(content) < 2:
        raise SystemExit(
            "Provide at least one --image or --pdf (OpenCode multimodal workaround).\n"
            "Example: scripts/vision_bridge.py 'Describe the plot' -i path/to.png"
        )

    out = _post_chat(
        args.base_url,
        args.api_key,
        args.model,
        content,
        args.max_tokens,
        args.temperature,
    )
    sys.stdout.write(out)
    if not out.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
