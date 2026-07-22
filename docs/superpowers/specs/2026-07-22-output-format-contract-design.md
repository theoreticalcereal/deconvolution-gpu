# Input and Output Format Contract Design

## Goal

Allow TIFF inputs to bypass pre-deconvolution normalization while honoring the
requested final output format, and keep OME-Zarr as the internal representation
for every non-TIFF input.

## Format matrix

| Input path | Requested format | Deconvolution path | Process output |
| --- | --- | --- | --- |
| TIFF | TIFF | Direct TIFF memory map and cuCIM | `DB2_*.tif` |
| TIFF | OZX | Direct TIFF memory map and cuCIM, then write OME-Zarr | `DB2_*.ozx` |
| Non-TIFF | TIFF | Normalize to OME-Zarr, stream cuCIM to OME-Zarr | `DB2_*.ozx`, then export TIFF |
| Non-TIFF | OZX | Normalize to OME-Zarr, stream cuCIM to OME-Zarr | `DB2_*.ozx` |

## Workflow contract

`output_formats` will be passed into `DECON` and `decon_wrapper.py`. The wrapper
will validate `tiff` or `ozx`. TIFF inputs will retain their direct deconvolution
path; only a requested OZX output will serialize the restored NumPy volume to
OME-Zarr for the existing archive step. Non-TIFF inputs will retain the current
OME-Zarr streaming path regardless of final format.

The `DECON` process output declaration will accept `DB2_*.tif`, `DB2_*.tiff`,
or `DB2_*.ozx`. When TIFF is requested, the existing export process will receive
either form: it copies a direct TIFF or converts an OZX to TIFF. OZX publishing
remains unchanged.

## Testing and recovery

Tests will cover all four matrix entries at the wiring/dispatch boundary and
assert that direct TIFF output satisfies the Nextflow process contract. The
completed TIFF from the failed run remains recoverable from its work directory;
the workflow task will not be rerun during implementation.
