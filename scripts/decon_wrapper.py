import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from pycudadecon import decon

def run_decon(image_path, psf_path, psf_file, background, iter_count):
    print("Running memory-safe Tiled GPU deconvolution...")
    
    full_psf_path = Path(psf_path) / psf_file
    
    # Load raw data and PSF
    image = imread(str(image_path))
    psf = imread(str(full_psf_path)).astype(np.float32)
    psf = np.abs(psf - background)
    
    # Setup tile sizing parameters
    tile_size = 512       # 512x512 blocks are highly optimized for GPU memory architectures
    overlap = 32          # Padded border to eliminate edge stitching artifacts
    
    nz, ny, nx = image.shape
    print(f"Original Volume Shape: Z={nz}, Y={ny}, X={nx}")
    
    # Allocate empty host memory for the final stitched output
    output = np.zeros_like(image, dtype=np.float32)
    
    # Loop over Y and X dimensions
    for y in range(0, ny, tile_size):
        for x in range(0, nx, tile_size):
            
            # Calculate block boundaries including the overlap padding
            y_start = max(0, y - overlap)
            y_end = min(ny, y + tile_size + overlap)
            x_start = max(0, x - overlap)
            x_end = min(nx, x + tile_size + overlap)
            
            # Extract the raw data tile
            tile = image[:, y_start:y_end, x_start:x_end]
            print(f"  Processing Tile at Y:{y}, X:{x} | Padded Shape: {tile.shape}...", end="", flush=True)
            
            # Deconvolve just this single tile on the GPU
            decon_tile = decon(tile, psf, n_iters=iter_count)
            
            # Calculate where to crop out the overlap padding
            crop_y_start = y - y_start
            crop_y_end = crop_y_start + min(tile_size, ny - y)
            crop_x_start = x - x_start
            crop_x_end = crop_x_start + min(tile_size, nx - x)
            
            # Target window inside our final reconstructed volume
            out_y_end = y + min(tile_size, ny - y)
            out_x_end = x + min(tile_size, nx - x)
            
            # Paste the clean cropped core into the output matrix
            output[:, y:out_y_end, x:out_x_end] = decon_tile[
                :, 
                crop_y_start:crop_y_end, 
                crop_x_start:crop_x_end
            ]
            print("Done.")

    # Normalize data stack and save out as a standard 16-bit TIFF
    print("Normalizing final volume...")
    output = (output - np.min(output)) / (np.max(output) - np.min(output) + 1e-6)
    output = (output * 65535).astype(np.uint16)
    
    output_filename = f"DB2_{Path(image_path).name}"
    imwrite(output_filename, output)
    print(f"Success! Budget saved. Output written to: {output_filename}")