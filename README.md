# Deep Learning Sound Classification Project

Sound classification using Convolutional Neural Networks for University of Bristol COMSM0018 Applied Deep Learning

* Author: Angus Redlarski Williams
* Github: @angusrw
* Email: angusrwilliams@gmail.com
* Co-contributors: Rachel Kirby (@rk16586)

---

Aim to recreate results from ["Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/), implementing Convolutional Neural Networks that use features from audio sample to classify original audio as one of 10 labels.

The project uses the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). The dataset is loaded in the following lines in [`main.py`](https://github.com/angusrw/cnn_sound_classification/blob/050267771480ac576bf662ba0b39412011d8a3c9/source/main.py#L54-L55) and [`main_with_adamw.py`](https://github.com/angusrw/cnn_sound_classification/blob/050267771480ac576bf662ba0b39412011d8a3c9/source/main_with_adamw.py#L54-L55):
```
train_data = UrbanSound8KDataset('UrbanSound8K_train.pkl', args.mode)
val_data = UrbanSound8KDataset('UrbanSound8K_test.pkl', args.mode)
```
`dataset.py` processes the dataset to extract relevant features.

The results of the project are visible in `report.pdf`.
