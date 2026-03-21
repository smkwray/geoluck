from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _copy_web_workspace(source_web: Path, preview_root: Path) -> None:
    if preview_root.exists():
        shutil.rmtree(preview_root)
    preview_root.mkdir(parents=True, exist_ok=True)

    for name in [
        "src",
        "public",
        "index.html",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "vite.config.ts",
    ]:
        source_path = source_web / name
        target_path = preview_root / name
        if source_path.is_dir():
            shutil.copytree(
                source_path,
                target_path,
                ignore=shutil.ignore_patterns("._*"),
            )
        else:
            shutil.copy2(source_path, target_path)

    node_modules = source_web / "node_modules"
    if not node_modules.exists():
        raise FileNotFoundError(f"Expected web dependencies at {node_modules}")
    os.symlink(node_modules, preview_root / "node_modules", target_is_directory=True)


def _build_preview(preview_root: Path) -> None:
    subprocess.run(["npm", "run", "build"], cwd=preview_root, check=True)


def _serve_dist(dist_dir: Path, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(dist_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving local geoluck preview at http://127.0.0.1:{port}/")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_web = repo_root / "web"
    preview_root = Path(tempfile.gettempdir()) / "geoluck-web-preview"

    _copy_web_workspace(source_web, preview_root)
    _build_preview(preview_root)
    if args.build_only:
        return
    _serve_dist(preview_root / "dist", args.port)


if __name__ == "__main__":
    main()
