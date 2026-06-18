# Deskew Process

The `DESKEW` process corrects oblique ctASLM acquisition geometry before
deconvolution. It is implemented by `workflow/scripts/deskew_wrapper.py` and
`workflow/scripts/deskew.m`.

## Inputs

Astrocyte handles file ingestion for package runs. Manual CLI runs can still
pass an existing `image_path` directory.

The process receives these values from Nextflow:

| Value | Meaning |
|---|---|
| `image_path` | Directory containing TIFFs, or a parent directory containing the legacy `cell_name` folder. |
| `cell_name` | Optional legacy dataset folder name under `image_path`. |
| `cell_index` | Optional value injected into MATLAB as `CellIndex`. |
| `dx` | Lateral pixel size in microns. |
| `dz` | Axial step size in microns. |
| `angle` | Light-sheet acquisition angle in degrees. |
| `flip` | Orientation flag used in the shear direction. |
| `output_dir` | Process-local output directory; Nextflow passes `.`. |

The MATLAB code looks for input TIFFs in:

```text
<image_path>/
```

When `cell_name` is supplied for a legacy folder layout, the MATLAB code looks
in:

```text
<image_path>/<cell_name>/
```

Supported raw input filenames are:

```text
CH00_000000.tif
CH00_000000.tiff
CH00_000000_registered_consistent.tif
CH00_000000_registered_consistent.tiff
```

The channel and timepoint scanner expects names matching:

```text
CH<two-or-more digits>_<six digits>[optional _registered_consistent].tif[f]
```

## Wrapper Behavior

`deskew_wrapper.py` builds one MATLAB `-batch` command. It injects only the
optional legacy variables that were actually supplied:

- `CellIndex` is omitted when `cell_index` is blank.

`deskew.m` discovers all channels and timepoints from the TIFF names.

## TIFF Reading

`readtiffstack.m` opens the TIFF with MATLAB's `Tiff` interface. It:

- Validates that the file exists.
- Reads stack dimensions from the first page.
- Preserves the numeric type from TIFF metadata.
- Fails if any page has a different size from the first page.

The stack is loaded as a MATLAB array with shape:

```text
rows x columns x z-slices
```

## Shear Correction

For each discovered channel/timepoint, `deskew.m` computes:

```text
newdz = dz * cosd(angle)
cz = floor(zsize / 2) + 1
yoffset = round(flip * (z - cz) * (newdz / dx))
```

Each Z plane is shifted along Y by `yoffset`. The output shear volume is padded
in Y by the largest possible offset so shifted planes fit without clipping.

After shearing, adjacent slices are averaged:

```text
ShearImage(:,:,z) = (ShearImage(:,:,z) + ShearImage(:,:,z+1)) / 2
```

This reduces striping/ringing in the top-view output.

## Top-View Rotation

The process then converts the sheared volume into the final top-view stack:

1. Compute `scale_x = dz * sind(angle) / dx`.
2. Resize the sheared volume along the third dimension with `imresize3`.
3. Rotate the scaled 3-D volume by `-flip * angle` around the Z axis with
   `imrotate3`.
4. Convert back to `uint16`.
5. Permute dimensions with `permute(rotTop_ShearImage, [1 3 2])`.

The result is written to `Top_shear`.

## Outputs

For each input `CH00_000000.tif`, the deskew process writes:

```text
shear/CH00_000000.tif
Top_shear/CH00_000000.tif
Top_shear/note.txt
```

`writetiffstack.m` writes one TIFF page per Z slice. If the estimated output
size is larger than 4 GiB, it uses BigTIFF mode automatically.

## Common Failure Points

`No TIFF files found` means the resolved input directory exists but contains no
`.tif` or `.tiff` files.

`No CH##_###### TIFF files found` means files exist, but they do not match the
expected raw naming pattern.

`Missing expected file` means channel/timepoint discovery or user filters asked
for a specific stack that was not present.
