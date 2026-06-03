import os
import sys

# Define the exact cluster path to the MATLAB runtime engine binaries
matlab_lib_path = "/home1/apps/MATLAB/R2022b/bin/glnxa64"

# 1. Force the current running process to recognize the path
if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = matlab_lib_path + ":" + os.environ["LD_LIBRARY_PATH"]
else:
    os.environ["LD_LIBRARY_PATH"] = matlab_lib_path

# 2. Tell Python's dynamic linker to look here (helps bypass SLURM stripping)
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(matlab_lib_path)
import matlab.engine
import argparse
from pathlib import Path

def run_decon(image_path, psf_path, psf_file, background, iter_count, output_dir):
    eng = matlab.engine.start_matlab()
    eng.addpath(str(Path(__file__).parent))
    eng.workspace['imagePath'] = image_path
    eng.workspace['psfPath'] = psf_path
    eng.workspace['psfFile'] = psf_file
    eng.workspace['background'] = float(background)
    eng.workspace['iter'] = int(iter_count)
    print(f"Running deconvolution with image: {image_path}, psf: {psf_path}/{psf_file}, background: {background}, iterations: {iter_count}")
    eng.run('blind_deconvolution.m', nargout=0)
    eng.quit()

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