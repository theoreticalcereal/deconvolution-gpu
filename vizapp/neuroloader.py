#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys
import threading
import urllib.parse


URL_PREFIX = r"/f"
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_RELATIVE_PATH = pathlib.Path("workflow/output/neuroglancer/layers.json")


class NeuroloaderError(Exception):
    pass


def load_layers_manifest(base_path=PACKAGE_ROOT, base_url=None):
    base = pathlib.Path(base_path).resolve()
    manifest_path = base / DEFAULT_MANIFEST_RELATIVE_PATH
    if not manifest_path.exists():
        raise NeuroloaderError(f"layers.json is missing: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise NeuroloaderError(f"layers.json contains invalid JSON: {manifest_path}") from exc

    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise NeuroloaderError(f"layers.json must contain a non-empty layers list: {manifest_path}")

    for layer in layers:
        if not isinstance(layer, dict):
            raise NeuroloaderError(f"Each layer entry must be an object in {manifest_path}")
        name = layer.get("name")
        source = layer.get("source")
        if not name or not source:
            raise NeuroloaderError(f"Each layer must contain name and source in {manifest_path}")
        source_path = validate_precomputed_source(source, manifest_path.parent, name)
        if base_url:
            layer["source"] = browser_precomputed_source(source_path, base, base_url)
    return layers


def validate_precomputed_source(source, published_dir, layer_name):
    prefix = "precomputed://file://"
    if not source.startswith(prefix):
        raise NeuroloaderError(f"Unsupported Neuroglancer source: {source}")

    manifest_path = pathlib.Path(source[len(prefix) :])
    published_path = pathlib.Path(published_dir) / layer_name
    path = published_path if (published_path / "info").is_file() else manifest_path
    if not path.exists() or not path.is_dir():
        raise NeuroloaderError(f"Layer source points to a missing precomputed dataset: {path}")
    if not (path / "info").is_file():
        raise NeuroloaderError(f"Layer source is missing precomputed info file: {path / 'info'}")
    return path.resolve()


def browser_precomputed_source(path, base_path, base_url):
    try:
        relative_path = pathlib.Path(path).resolve().relative_to(base_path)
    except ValueError as exc:
        raise NeuroloaderError(f"Layer source is outside the served VizApp directory: {path}") from exc

    quoted_path = urllib.parse.quote(relative_path.as_posix())
    return f"precomputed://{base_url.rstrip('/')}{URL_PREFIX}/{quoted_path}"


def create_viewer(layers, port, base_path=PACKAGE_ROOT):
    try:
        import neuroglancer
        import tornado.web as web
    except ImportError as exc:
        raise NeuroloaderError(f"Missing VizApp dependency: {exc}") from exc

    neuroglancer.set_server_bind_address(bind_address="0.0.0.0", bind_port=port)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as state:
        for layer in layers:
            state.layers.append(
                name=layer["name"],
                layer=neuroglancer.ImageLayer(source=layer["source"]),
            )

    neuroglancer.server.global_server.app.add_handlers(
        r".*$",
        [(URL_PREFIX + r"/(.*)", web.StaticFileHandler, {"path": str(base_path)})],
    )
    neuroglancer.server.global_server.app.add_handlers(
        r".*$",
        [
            (
                "/",
                web.RedirectHandler,
                {"url": urllib.parse.urlparse(str(viewer)).path},
            )
        ],
    )
    return viewer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Load generated Neuroglancer layers.")
    parser.add_argument("vizapp_port", type=int, help="VizApp port number")
    return parser.parse_args(argv)


def stay_running():
    stop_event = threading.Event()
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        print("Stopping viewer...")


def main(argv=None):
    args = parse_args(argv)
    base_url = f"http://127.0.0.1:{args.vizapp_port}"
    try:
        layers = load_layers_manifest(PACKAGE_ROOT, base_url)
        viewer = create_viewer(layers, args.vizapp_port, PACKAGE_ROOT)
    except NeuroloaderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"vizapp port: {args.vizapp_port}")
    print(f"Folder: {PACKAGE_ROOT}")
    print(f"Loaded {len(layers)} Neuroglancer layer(s)")
    print(viewer)
    return 0


if __name__ == "__main__":
    if main() == 0:
        stay_running()
    else:
        sys.exit(1)
