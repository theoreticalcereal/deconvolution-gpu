#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import sys
import threading
import urllib.parse


URL_PREFIX = r"/f"
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_RELATIVE_PATH = pathlib.Path("workflow/output/neuroglancer/layers.json")


class NeuroloaderError(Exception):
    pass


def resolve_manifest_path(base_path):
    override = os.environ.get("NEUROGLANCER_MANIFEST")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    return pathlib.Path(base_path) / DEFAULT_MANIFEST_RELATIVE_PATH


def load_layers_manifest(base_path=PACKAGE_ROOT):
    manifest_path = resolve_manifest_path(base_path)
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
        validate_precomputed_source(source)
    return layers


def validate_precomputed_source(source):
    prefix = "precomputed://file://"
    if not source.startswith(prefix):
        raise NeuroloaderError(f"Unsupported Neuroglancer source: {source}")

    path = pathlib.Path(source[len(prefix) :])
    if not path.exists() or not path.is_dir():
        raise NeuroloaderError(f"Layer source points to a missing precomputed dataset: {path}")
    if not (path / "info").is_file():
        raise NeuroloaderError(f"Layer source is missing precomputed info file: {path / 'info'}")


def create_viewer(layers, port):
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
        [(URL_PREFIX + r"/(.*)", web.StaticFileHandler, {"path": str(PACKAGE_ROOT)})],
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
    try:
        layers = load_layers_manifest(PACKAGE_ROOT)
        viewer = create_viewer(layers, args.vizapp_port)
    except NeuroloaderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"vizapp port: {args.vizapp_port}")
    print(f"Loaded {len(layers)} Neuroglancer layer(s)")
    print(viewer)
    return 0


if __name__ == "__main__":
    if main() == 0:
        stay_running()
    else:
        sys.exit(1)
