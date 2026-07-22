# GPU Memory Cleanup and PSF Tile Selection Design

## Goal

Make chunked cuCIM restoration reliable on an 8 GiB GPU and reduce native
CuPy blind-PSF estimation time without changing its numerical update equations.
The accelerated PSF and restored-volume outputs must remain compatible with the
comparison suite under `workflow/scripts`.

## GPU memory lifecycle

`deconvolve_with_cucim` owns each restoration chunk's CuPy arrays. After the
restored array has been copied to host memory, it will synchronize the device,
drop all live chunk arrays, clear the device's cuFFT plan cache, and release the
default CuPy device-memory pool. Cleanup will run in a `finally` block so an OOM
or another restoration error cannot leave cached allocations behind for a later
chunk. The cleanup policy favors predictable operation on 8 GiB devices over
retaining allocations between chunks.

## Representative blind-PSF tiles

Blind estimation will gain a `blind_max_tiles` integer parameter. Its default
will be 16, while 0 means the current full tile grid. The existing tile grid and
VRAM-aware tile dimensions remain unchanged.

When the candidate grid exceeds the limit, the selector will score every tile
with the existing SNR measure and choose a deterministic, spatially distributed
subset. It will partition the XY tile grid into balanced regions and select the
highest-SNR candidate within each region, then use globally strongest remaining
candidates if a non-square limit leaves slots. Stable coordinate ordering will
break equal-score ties. Selection will not alter a chosen tile's halo, Z window,
blind-RL iteration count, PSF normalization, or SNR merge weight.

Logs will report the candidate and selected counts, selected coordinates and
scores, and the reduction in blind-RL work. The cache key will include the tile
limit and selection strategy so a 16-tile PSF cannot be confused with a
full-grid PSF.

## Parameter wiring

`blind_max_tiles` will be exposed through:

- `psf_estimation.py` and `decon_wrapper.py` command-line interfaces;
- `workflow/modules.nf` and `workflow/configs/nextflow.config`;
- `astrocyte_pkg.yml` with a default of 16 and minimum of 0;
- `params.yml` for the current comparison run, set to 16;
- operator documentation.

Setting `blind_max_tiles: 0` in a comparison run will reproduce full-grid tile
selection. Existing cache data remains separated because the selection settings
are part of the cache key.

## Testing and handoff

Tests will first establish failures for deterministic distributed selection,
full-grid compatibility, cache separation, parameter propagation, and GPU
cleanup on success and exception paths. The minimal implementation will then be
added and the focused and repository test suites run.

No Nextflow task will be launched as part of implementation. The user will run
the accelerated and full-grid workflows and evaluate their PSFs with the
comparison scripts.
