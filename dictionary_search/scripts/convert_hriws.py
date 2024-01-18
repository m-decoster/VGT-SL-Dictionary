import argparse
import glob
import json
import os


def convert(input_path, output_path):
    input_files = glob.glob(os.path.join(input_path, '*.npy'))
    for input_file in input_files:
        basename = os.path.basename(input_file)
        gt = basename.split('_')[1].split('.')[0]
        basename_wo_extension, _ext = os.path.splitext(basename)
        with open(os.path.join(output_path, basename_wo_extension + '.json'), 'w') as f:
            json.dump({'ground_truth': gt, 'keypoints_file': basename_wo_extension, 'predictions': []}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_directory', type=str, help='The path to the directory containing the query videos.')
    parser.add_argument('output_directory', type=str,
                        help='The output directory.')

    args = parser.parse_args()

    convert(args.input_directory, args.output_directory)
