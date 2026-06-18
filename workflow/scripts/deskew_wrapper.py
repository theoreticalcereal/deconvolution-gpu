import argparse
import subprocess
from pathlib import Path
import sys


def run_deskew(image_path, cell_name, cell_index, dx, dz, angle, flip, output_dir):

    script_dir = str(Path(__file__).parent.absolute())
    cell_name = "" if cell_name is None else str(cell_name).strip()

    print(f"Running deskew with image: {image_path}, cell name: {cell_name}, "
          f"cell index: {cell_index!r}, dx: {dx}, dz: {dz}, "
          f"angle: {angle}, flip: {flip}, output_dir: {output_dir}")

    # Only set CellIndex if a non-empty value was provided.
    cell_index_line = ""
    if cell_index and str(cell_index).strip():
        cell_index_line = f"CellIndex=int32({cell_index}); "

    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"imagePath='{image_path}'; "
        f"CellName='{cell_name}'; "
        + cell_index_line
        + f"dx={dx}; "
        f"dz={dz}; "
        f"angle={angle}; "
        f"flip={flip}; "
        f"output_dir='{output_dir}'; "
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
    parser.add_argument('--cell_name', default='')
    parser.add_argument('--cell_index', default='')
    parser.add_argument('--dx')
    parser.add_argument('--dz')
    parser.add_argument('--angle')
    parser.add_argument('--flip')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    run_deskew(args.image_path, args.cell_name, args.cell_index,
               args.dx, args.dz, args.angle, args.flip, args.output_dir)
