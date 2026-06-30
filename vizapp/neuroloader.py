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
        if base_url:
            layer["source"] = rewrite_local_zarr_source(source, manifest_path.parent, name, base, base_url)
        else:
            validate_ome_zarr_source(source, manifest_path.parent, name)
    return layers


def rewrite_local_zarr_source(source, published_dir, layer_name, base_path, base_url):
    source_path = validate_ome_zarr_source(source, published_dir, layer_name)
    return browser_zarr_source(source_path, pathlib.Path(base_path).resolve(), base_url)


def validate_ome_zarr_source(source, published_dir, layer_name):
    prefix = "zarr://file://"
    if source.startswith("precomputed://"):
        raise NeuroloaderError("precomputed sources are no longer supported; expected zarr://file:// OME-Zarr sources")
    if not source.startswith(prefix):
        raise NeuroloaderError(f"Unsupported Neuroglancer source: {source}")

    manifest_path = zarr_file_source_path(source)
    published_path = published_ome_zarr_path(published_dir, layer_name)
    path = published_path if (published_path / ".zattrs").is_file() else manifest_path
    if not path.exists() or not path.is_dir():
        raise NeuroloaderError(f"Layer source points to a missing OME-Zarr dataset: {path}")
    if path.name.endswith(".ome.zarr") is False:
        raise NeuroloaderError(f"Layer source must point to a .ome.zarr directory: {path}")
    if not (path / ".zattrs").is_file() or not (path / ".zgroup").is_file():
        raise NeuroloaderError(f"Layer source is missing OME-Zarr metadata: {path}")
    if not (path / "0" / ".zarray").is_file():
        raise NeuroloaderError(f"Layer source is missing OME-Zarr scale 0 array metadata: {path / '0' / '.zarray'}")
    validate_ome_zarr_metadata(path)
    return path.resolve()


def published_ome_zarr_path(published_dir, layer_name):
    neuroglancer_dir = pathlib.Path(published_dir)
    output_dir = neuroglancer_dir.parent
    deconvolved_path = output_dir / "deconvolved" / f"{layer_name}.ome.zarr"
    if (deconvolved_path / ".zattrs").is_file():
        return deconvolved_path
    return neuroglancer_dir / f"{layer_name}.ome.zarr"


def validate_ome_zarr_metadata(path):
    try:
        zattrs = json.loads((path / ".zattrs").read_text())
        zgroup = json.loads((path / ".zgroup").read_text())
        zarray = json.loads((path / "0" / ".zarray").read_text())
    except json.JSONDecodeError as exc:
        raise NeuroloaderError(f"Layer source contains invalid OME-Zarr JSON metadata: {path}") from exc

    if zgroup.get("zarr_format") != 2 or zarray.get("zarr_format") != 2:
        raise NeuroloaderError(f"Layer source contains invalid OME-Zarr Zarr v2 metadata: {path}")

    multiscales = zattrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise NeuroloaderError(f"Layer source contains invalid OME-Zarr multiscales metadata: {path}")

    datasets = multiscales[0].get("datasets") if isinstance(multiscales[0], dict) else None
    if not isinstance(datasets, list) or not datasets or datasets[0].get("path") != "0":
        raise NeuroloaderError(f"Layer source contains invalid OME-Zarr multiscales metadata: {path}")


def zarr_file_source_path(source):
    parsed = urllib.parse.urlparse(source[len("zarr://") :])
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise NeuroloaderError(f"Unsupported Neuroglancer source: {source}")
    return pathlib.Path(urllib.parse.unquote(parsed.path))


def browser_zarr_source(path, base_path, base_url):
    try:
        relative_path = pathlib.Path(path).resolve().relative_to(base_path)
    except ValueError as exc:
        raise NeuroloaderError(f"Layer source is outside the served VizApp directory: {path}") from exc

    quoted_path = urllib.parse.quote(relative_path.as_posix())
    return f"zarr://{base_url.rstrip('/')}{URL_PREFIX}/{quoted_path}"


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
