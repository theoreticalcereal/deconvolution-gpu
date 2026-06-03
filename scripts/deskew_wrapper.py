import argparse
import subprocess
from pathlib import Path
import sys

def run_deskew(image_path, cell_name, cell_index, channels, 
               timepoints, dx, dz, angle, flip, output_dir):
    
    script_dir = str(Path(__file__).parent.absolute())
    
    print(f"Running deskew with image: {image_path}, cell name: {cell_name}, cell index: {cell_index}, channels: {channels}, timepoints: {timepoints}, dx: {dx}, dz: {dz}, angle: {angle}, flip: {flip}")

    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"imagePath='{image_path}'; "
        f"CellName='{cell_name}'; "
        f"CellIndex=int32({cell_index}); "
        f"dx={dx}; "
        f"dz={dz}; "
        f"angle={angle}; "
        f"flip={flip}; "
        f"run('deskew.m');"
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
    parser.add_argument('--cell_name')
    parser.add_argument('--cell_index')
    parser.add_argument('--channels')
    parser.add_argument('--timepoints')
    parser.add_argument('--dx')
    parser.add_argument('--dz')
    parser.add_argument('--angle')
    parser.add_argument('--flip')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    
    run_deskew(args.image_path, args.cell_name, args.cell_index, args.channels,
               args.timepoints, args.dx, args.dz, args.angle, args.flip, args.output_dir)