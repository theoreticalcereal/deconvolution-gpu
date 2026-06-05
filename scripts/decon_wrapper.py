# dask-orchestrated gpu deconvolution wrapper
# designed for single-gpu node testing before scaling to cluster

import argparse
from pathlib import Path
import numpy as np
import dask.array as da
from tifffile import imread, imwrite
from pycudadecon import decon
import psfmodels as pm

def generate_theoretical_psf(na, wavelength, ni, dxy, dz, background):
    psf = pm.make_profile(
        model="gibsonlanni", na=na, wavelength=wavelength, ni=ni,
        res_lateral=dxy, res_axial=dz, size_x=128, size_y=128, size_z=61
    ).astype(np.float32)
    return np.abs(psf - background)

def decon_worker(chunk, psf, n_iters):
    """
    Dask sends perfectly padded numpy chunks here.
    We just process them and return the array.
    """
    processed = decon(chunk, psf, n_iters=n_iters)
    return np.clip(processed, 0, 65535).astype(np.uint16)

def main():
    parser = argparse.ArgumentParser(description="Dask-orchestrated GPU deconvolution")
    parser.add_argument('--image_path', required=True, help="path to deskewed input TIFF")
    parser.add_argument('--background', type=float, default=0.0, help="background subtraction value")
    parser.add_argument('--iter', type=int, default=10, help="number of Lucy Richardson iterations")
    
    # optical parameters
    parser.add_argument('--na', type=float, default=1.0)
    parser.add_argument('--wavelength', type=float, default=0.525)
    parser.add_argument('--ni', type=float, default=1.33)
    parser.add_argument('--dxy', type=float, default=1.0)
    parser.add_argument('--dz', type=float, default=1.0)
    args = parser.parse_args()

    # 1. Setup PSF (Theoretical for now, eventually replaced by your extracted blind PSF)
    psf = generate_theoretical_psf(
        na=args.na, wavelength=args.wavelength, ni=args.ni, 
        dxy=args.dxy, dz=args.dz, background=args.background
    )

    # 2. Lazy Load the Data with Dask
    # We specify the exact chunk size that we know the Tesla P4 can handle
    image_array = imread(args.image_path)
    nz, ny, nx = image_array.shape
    lazy_image = da.from_array(image_array, chunks=(nz, 256, 256))

    print(f"Loaded {args.image_path} into Dask graph. Executing map_overlap...", flush=True)

    # 3. Dask Orchestration
    # map_overlap handles all the padding, cropping, and stitching invisibly
    processed_lazy = lazy_image.map_overlap(
        decon_worker,
        depth={0: 0, 1: 32, 2: 32},  # 0 padding in Z, 32px padding in Y and X
        boundary='reflect',          # mirror the edges to prevent dark borders
        dtype=np.uint16,
        psf=psf,
        n_iters=args.iter
    )

    # 4. Compute and Save
    # The .compute() command actually fires the GPU processing graph
    output = processed_lazy.compute(scheduler='single-threaded') 
    
    raw_stem = Path(args.image_path).name.replace(".tiff", "").replace(".tif", "")
    output_filename = f"DB2_{raw_stem}.tif" if "CH" in raw_stem else "DB2_deconvolved_output.tif"
    imwrite(Path(output_filename).name, output)
    print("Dask processing complete and saved!", flush=True)

if __name__ == '__main__':
    main()