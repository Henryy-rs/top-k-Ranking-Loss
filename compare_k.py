import argparse
import os
import torch
import torch.backends.cudnn as cudnn

from network.TorchUtils import TorchModel
from features_loader import FeaturesLoaderVal
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from os import path
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Compare k and draw graph.")
    parser.add_argument('--r', type=str, default="exps/models/r3d",
                        help="r3d models path")
    parser.add_argument('--i', type=str, default="exps/models/i3d",
                        help="i3d models path")
    parser.add_argument('--c', type=str, default="exps/models/c3d",
                        help="c3d models path")
    parser.add_argument('--annotation_path', default="Test_Annotation.txt",
                        help="path to annotations")
    parser.add_argument('--know_best', default=False,
                        help="path to annotations")

    return parser.parse_args()


def validate(model, data_iter):
    cudnn.benchmark = True

    y_trues = torch.tensor([])
    y_preds = torch.tensor([])

    with torch.no_grad():
        for features, start_end_couples, lengths in tqdm(data_iter):
            # features is a batch where each item is a tensor of 32 4096D features
            features = features.to(device)
            outputs = model(features).squeeze(-1)  # (batch_size, 32)
            for vid_len, couples, output in zip(lengths, start_end_couples, outputs.cpu().numpy()):
                y_true = np.zeros(vid_len)
                y_pred = np.zeros(vid_len)

                segments_len = vid_len // 32
                for couple in couples:
                    if couple[0] != -1:
                        y_true[couple[0]: couple[1]] = 1

                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = output[i]

                if y_trues is None:
                    y_trues = y_true
                    y_preds = y_pred
                else:
                    y_trues = np.concatenate([y_trues, y_true])
                    y_preds = np.concatenate([y_preds, y_pred])

    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_max_auc(models_path, data_iter, early_stop_count=4, best_epoch=None):
    max_auc = 0
    max_epoch = 0
    if best_epoch:
        model = TorchModel.load_model(models_path).to(device).eval()
        max_auc = validate(model, data_iter)
        return max_epoch, max_auc

    for epoch, pt in enumerate(sorted(list(os.listdir(models_path)), key=lambda x: int(x.split('_')[1].split('.')[0]))):
        if early_stop_count == 0:
            print("break")
            break
        model_path = os.path.join(models_path, pt)
        print(pt)
        model = TorchModel.load_model(model_path).to(device).eval()
        roc_auc = validate(model, data_iter)
        print(roc_auc)

        if roc_auc > max_auc:
            max_auc = roc_auc
            max_epoch = epoch
        else:
            early_stop_count -= 1

    return max_epoch + 1, max_auc


def get_dots2plot(feature_dim, feature_path, extractor_path, know_best=False):
    data_loader = FeaturesLoaderVal(features_path=feature_path,
                                    feature_dim=feature_dim,
                                    annotation_path="Test_Annotation.txt")

    data_iter = torch.utils.data.DataLoader(data_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,  # 4, # change this part accordingly
                                            pin_memory=True)
    best_file = extractor_path.split('/')[-1] + ".txt"
    x = []
    y = []

    if know_best:
        with open(best_file, 'r') as f:
            for k, k_path in enumerate(os.listdir(extractor_path)):
                best_epoch = int(f.readline().split(' ')[0]) - 1
                epochs_path = os.path.join(extractor_path, "k" + str(k+1), "epoch_" + str(best_epoch) + ".pt")
                max_epoch, max_auc = get_max_auc(epochs_path, data_iter, best_epoch=best_epoch)
                x.append(k + 1)
                y.append(max_auc)

    else:
        with open(best_file, 'w') as f:
            for k, k_path in enumerate(os.listdir(extractor_path)):
                epochs_path = os.path.join(extractor_path, k_path)
                max_epoch, max_auc = get_max_auc(epochs_path, data_iter)
                f.write(' '.join([str(max_epoch), str(max_auc)]) + "\n")
                x.append(k + 1)
                y.append(max_auc)
    return x, y


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure()
    lw = 1.05
    ax = plt.axes()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    plt.grid(True)

    plt.plot(*get_dots2plot(2048, "features", args.r, know_best=args.know_best), lw=lw,  marker='o',  linestyle='dashed', label='R3D')
    plt.plot(*get_dots2plot(1024, "features_i3d", args.i, know_best=args.know_best), lw=lw,  marker='o',  linestyle='dashed', label='I3D')
    plt.plot(*get_dots2plot(4096, "features_c3d", args.c,  know_best=args.know_best), lw=lw,  marker='o',  linestyle='dashed', label='C3D')
    plt.xlabel('k')
    plt.ylabel('AUC')
    plt.legend(loc="lower right")
    plt.savefig(path.join('graphs', 'compare_k.png'))
    plt.close()
