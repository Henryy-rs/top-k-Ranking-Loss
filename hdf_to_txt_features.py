import argparse
import h5py
import os


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
    parser.add_argument('--hdf_path', default='./features_hdf5/charades_i3d_rgb_stride_1s.hdf5',
                        help="path to hdf features")
    parser.add_argument('--export_path', type=str, default="./features_i3d",
                        help="set logging file.")
    return parser.parse_args()


def main():
    args = get_args()
    f = h5py.File(args.hdf_path, 'r')
    for key in f:
        video_class_path = os.path.join(args.export_path, key)
        if not os.path.exists(video_class_path):
            os.mkdir(video_class_path)
        for video in f[key]:
            feature_path = os.path.join(video_class_path, video.split('.')[0] + ".txt")
            with open(feature_path, 'w') as feature_file:
                feature = f[key][video][list(f[key][video])[0]]
                for line in feature:
                    line = [str(value) for value in line]
                    feature_file.write(' '.join(line) + '\n')
    f.close()


if __name__=='__main__':
    main()