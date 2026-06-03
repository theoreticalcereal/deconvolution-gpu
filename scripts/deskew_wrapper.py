import matlab.engine
import argparse
from pathlib import Path
import sys

def run_deskew(image_path, cell_name, cell_index, channels, 
               timepoints, dx, dz, angle, flip, output_dir):
    eng = matlab.engine.start_matlab()
    eng.addpath(str(Path(__file__).parent))
    eng.workspace['imagePath'] = image_path
    eng.workspace['CellName'] = cell_name
    eng.workspace['CellIndex'] = matlab.int32([cell_index])
    eng.workspace['dx'] = dx
    eng.workspace['dz'] = dz
    eng.workspace['angle'] = angle
    eng.workspace['flip'] = flip
    print(f"Running deskew with image: {image_path}, cell name: {cell_name}, cell index: {cell_index}, channels: {channels}, timepoints: {timepoints}, dx: {dx}, dz: {dz}, angle: {angle}, flip: {flip}")
    eng.run('deskew.m', nargout=0)
    eng.quit()

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