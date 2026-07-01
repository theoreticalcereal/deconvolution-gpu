# Neuroglancer VizApp

The workflow writes OME-Zarr image outputs under `workflow/output/deconvolved/`
and a Neuroglancer manifest at `workflow/output/neuroglancer/layers.json`.
Launch the VizApp after the workflow has completed and those files exist.

## Launch from Astrocyte

Run the workflow first, then launch the Neuroglancer VizApp from the Astrocyte
workflow page. Astrocyte sets `VIZAPP_PORT` and runs
`vizapp/run_neuroglancer.sh`.

The VizApp loads the BioHPC `neuroglancer/2.40.1` module, starts
`vizapp/neuroloader.py`, validates the OME-Zarr metadata, and serves the local
Zarr files to the browser.

## Launch Manually

From the package root:

```bash
test -f workflow/output/neuroglancer/layers.json
cd vizapp
export VIZAPP_PORT=9876
./run_neuroglancer.sh
```

Open the printed Neuroglancer URL. For local testing, this is usually:

```text
http://127.0.0.1:9876
```

## Display Defaults

The VizApp applies a grayscale shader to every image layer with the normalized
contrast range set to `0` through `400`. This makes the initial view use 400 as
the white point instead of stretching to the data maximum. The shader control
remains available in Neuroglancer if you need to adjust contrast interactively.

## Troubleshooting

If the VizApp reports that `layers.json` is missing, confirm that the workflow
published outputs under `workflow/output/`. If you used a different
`--output_dir`, relaunch with the default output directory or place the
`neuroglancer/` and `deconvolved/` outputs under `workflow/output/`.

If Neuroglancer reports an OME-Zarr metadata error, regenerate outputs with the
current workflow code. Older outputs may contain metadata that newer
Neuroglancer builds reject.
