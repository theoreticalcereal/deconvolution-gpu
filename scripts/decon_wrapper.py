import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from pycudadecon import decon

def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated Tiled Deconvolution")
    parser.add_argument('--image_path', required=True, help="Path to deskewed input TIFF")
    parser.add_argument('--psf_path', required=True, help="Directory containing PSF")
    parser.add_argument('--psf_file', required=True, help="PSF filename")
    parser.add_argument('--background', type=float, default=0.0, help="Background subtraction value")
    parser.add_argument('--iter', type=int, default=10, help="Number of Richardson-Lucy iterations")
    args = parser.parse_args()

    # load Data
    image = imread(args.image_path)
    psf_full_path = Path(args.psf_path) / args.psf_file
    psf = imread(str(psf_full_path)).astype(np.float32)
    psf = np.abs(psf - args.background)
    
    nz, ny, nx = image.shape

    # setup tiling arrays
    tile_size = 256
    overlap = 32
    output = None  # initiated as none, allocated dynamically on the first tile output

    # tiling execution loop
    for y in range(0, ny, tile_size):
        for x in range(0, nx, tile_size):
            # calculate padded block boundaries
            y_start = max(0, y - overlap)
            y_end = min(ny, y + tile_size + overlap)
            x_start = max(0, x - overlap)
            x_end = min(nx, x + tile_size + overlap)
            
            # slice and execute
            tile = image[:, y_start:y_end, x_start:x_end]
            decon_tile = decon(tile, psf, n_iters=args.iter)
            print(f"Finished Tile Y:{y}-{y_end} X:{x}-{x_end}... ({np.round((y*nx + x)/(ny*nx)*100, 1)}% complete)", flush=True)
            
            # dynamic allocation based on actual GPU output Z-slices
            if output is None:
                output = np.zeros((decon_tile.shape[0], ny, nx), dtype=np.uint16)
            
            # calculate crop margins to eliminate edge artifacts
            crop_y_start = y - y_start
            crop_y_end = crop_y_start + min(tile_size, ny - y)
            crop_x_start = x - x_start
            crop_x_end = crop_x_start + min(tile_size, nx - x)
            
            # calculate target window in the final volume
            tile_nz, tile_ny, tile_nx = cleaned_tile.shape
            out_y_end = y + tile_ny
            out_x_end = x + tile_nx
            
            # clip limits, cast to uint16, and insert into output matrix
            output[:, y:out_y_end, x:out_x_end] = np.clip(
                decon_tile[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end], 
                0, 65535
            ).astype(np.uint16)

    # hopefully bulletproof local file saving
    raw_stem = Path(args.image_path).name.replace(".tiff", "").replace(".tif", "")
    
    if not raw_stem or "CH" not in raw_stem:
        output_filename = "DB2_deconvolved_output.tif"
    else:
        output_filename = f"DB2_{raw_stem}.tif"
        
    # strip any rogue directory paths so it writes exactly to Nextflow's work dir
    local_output_path = Path(output_filename).name 
    imwrite(local_output_path, output)

if __name__ == '__main__':
    main()