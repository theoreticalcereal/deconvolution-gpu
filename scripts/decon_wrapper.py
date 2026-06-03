import argparse
import subprocess
import sys
from pathlib import Path

def run_decon(image_path, psf_path, psf_file, background, iter_count, output_dir):
    
    script_dir = str(Path(__file__).parent.absolute())
    
    print(f"Running deconvolution with image: {image_path}, psf: {psf_path}/{psf_file}, background: {background}, iterations: {iter_count}")

    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"imagePath='{image_path}'; "
        f"psfPath='{psf_path}'; "
        f"psfFile='{psf_file}'; "
        f"background={background}; "
        f"iter={iter_count}; "
        f"run('blind_deconvolution.m');"
    )

    command = ["matlab", "-batch", matlab_cmd]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"MATLAB execution failed with error code: {e.returncode}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--psf_path')
    parser.add_argument('--psf_file')
    parser.add_argument('--background', type=float)
    parser.add_argument('--iter', type=int)
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    
    run_decon(args.image_path, args.psf_path, args.psf_file,
              args.background, args.iter, args.output_dir)