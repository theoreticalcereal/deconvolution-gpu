import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from pycudadecon import decon

def run_decon(image_path, psf_path, psf_file, background, iter_count):
    print("Running GPU deconvolution with provided PSF...")
    
    full_psf_path = Path(psf_path) / psf_file
    
    # Load data
    image = imread(str(image_path))
    psf = imread(str(full_psf_path)).astype(np.float32)
    
    # Apply background subtraction to your provided PSF
    psf = psf - background
    psf = np.abs(psf)
    
    # GPU Richardson-Lucy deconvolution
    deconvolved = decon(
        image, 
        psf, 
        n_iters=iter_count,
        clip_negative=True,
        bandwidth=0.5,       # standard for fluorescence
        remove_nans=True,
    )
    
    # Normalize and convert to 16-bit
    deconvolved = (deconvolved - np.min(deconvolved)) / \
                  (np.max(deconvolved) - np.min(deconvolved) + 1e-6)
    deconvolved = (deconvolved * 65535).astype(np.uint16)
    
    # Save locally for Nextflow to publish
    output_filename = f"DB2_{Path(image_path).name}"
    imwrite(output_filename, deconvolved)
    print(f"Done! Saved as: {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--psf_path', required=True)
    parser.add_argument('--psf_file', required=True)
    parser.add_argument('--background', type=float, default=0.0)
    parser.add_argument('--iter', type=int, default=10)
    args = parser.parse_args()
    
    run_decon(args.image_path, args.psf_path, args.psf_file, args.background, args.iter)