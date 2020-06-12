import torch
import torchvision
from torch.nn import functional as F
import numpy as np
from typing import Union, NamedTuple

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main():
    lmc_logits = torch.load("LMC.pt")
    mc_logits = torch.load("MC.pt")
    class_labels = torch.load("labels.pt")
    filenames = torch.load("files.pt")
    tscnn_logits = torch.Tensor()

    lmc_scores = F.softmax(lmc_logits[i, :])
    mc_scores = F.softmax(mc_logits[i, :])
    tscnn_scores = (lmc_scores + mc_scores) / 2

    results = {"preds":[], "labels":[]}

    file_logit_dict = {}
    file_label_dict = {}

    for i in range(0,len(filenames)):
        if filenames[i] in file_logit_dict:
            file_logit_dict[filenames[i]] += tscnn_scores[i,:]
        else:
            file_logit_dict[filenames[i]] = tscnn_scores[i,:]
            file_label_dict[filenames[i]] = class_labels[i]

    for key,val in file_logit_dict.items():
        pred = val.argmax(dim=-1)
        results["preds"].append(pred)
        results["labels"].append(file_label_dict[key])

    #pca = compute_pca(class_labels, file_labels, final_scores)

    pca = compute_pca(
        np.array(results["labels"]), np.array(results["preds"])
    )

    total = 0
    for key, val in pca.items():
        total += val
        print(f"Class {key} accuracy: {val*100:2.2f}")

    pca_avg = total/10
    print(f"Average accuracy: {pca_avg}")


def compute_pca(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
):
    assert len(labels) == len(preds)

    # stores total number of examples for each class
    total = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    # stores total number of correct predictions for each class
    correct = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    # stores accuracy for each class
    pca = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0, 9:0.0}

    for i in range(0,len(labels)-1):
        total[labels[i]] += 1
        if labels[i] == preds[i]:
            correct[labels[i]] += 1

    for key, val in pca.items():
        pca[key] = (correct[key]/total[key])

    return pca

if __name__ == "__main__":
    main()
