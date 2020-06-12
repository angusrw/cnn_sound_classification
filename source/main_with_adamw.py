import time
import argparse
from pathlib import Path

from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torchvision
import torchvision.datasets
import torch.backends.cudnn
import numpy as np
from torchvision.transforms import ToTensor
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import UrbanSound8KDataset

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train CNN on UrbanSound8K dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--mode", default="LMC", type=str, help="Which feature mode to execute for (MC, LMC or MLMC)")
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout variable")
parser.add_argument("--batch-size", default=32, type=int, help="Number of samples within each mini-batch")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
parser.add_argument("--workers", default=8, type=int, help="Number of workers for loaders")
parser.add_argument("--decay", default=1e-4, type=float, help="Weight decay to use in SGD Optimizer")
parser.add_argument("--momentum", default=0.9, type=float, help="Learning rate")
parser.add_argument("--val-frequency", default=10, type=int, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--log-frequency", default=10, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-frequency", default=10, type=int, help="How frequently to print progress to the command line in number of steps")

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = ToTensor()

    train_data = UrbanSound8KDataset('UrbanSound8K_train.pkl', args.mode)
    val_data = UrbanSound8KDataset('UrbanSound8K_test.pkl', args.mode)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout, mode=args.mode)
    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay, amsgrad=False)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader,  val_loader, criterion, optimizer, summary_writer, DEVICE, args.mode
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float, mode: str):
        super().__init__()

        if (mode=="MLMC"): self.input_shape = ImageShape(height=145, width=41, channels=channels)
        else: self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.drop1D = nn.Dropout(dropout)
        self.drop2D = nn.Dropout2d(dropout)

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1)

        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn64_1 = nn.BatchNorm2d(64)
        self.bn64_2 = nn.BatchNorm2d(64)
        self.bnfc = nn.BatchNorm1d(1024)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1,
        )

        fc_in = 15488
        if (mode=="MLMC"):fc_in = 26048
        self.fc1 = nn.Linear(fc_in, 1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.conv3)
        self.initialise_layer(self.conv4)
        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn32_1(self.conv1(input)))
        x = self.drop2D(F.relu(self.bn32_2(self.conv2(x))))
        x = self.pool1(x)
        x = F.relu(self.bn64_1(self.conv3(x)))
        x = self.drop2D(F.relu(self.bn64_2(self.conv4(x))))
        x = self.pool2(x)
        x = torch.flatten(x,1)
        x = self.drop1D(torch.sigmoid(self.bnfc(self.fc1(x))))
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        mode: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.mode = mode

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for i, (input, target, filename) in enumerate(self.train_loader):
                batch = input.to(self.device)
                labels = target.to(self.device)
                data_load_end_time = time.time()
                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate(epoch, epochs)
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self, epoch, epochs):
        results = {"preds": [], "labels": []}
        class_labels = []
        filenames = []
        final_logits = torch.Tensor()
        final_logits = final_logits.to(self.device)
        total_loss = 0

        self.model.eval()

        with torch.no_grad():
            for i, (input, target, filename) in enumerate(self.val_loader):
                batch = input.to(self.device)
                labels = target.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                class_labels.extend(list(labels.cpu().numpy()))
                filenames.extend(list(filename))

                final_logits = torch.cat([final_logits, logits], dim=0)

        file_logit_dict = {}
        file_label_dict = {}

        for i in range(0,len(filenames)):
            if filenames[i] in file_logit_dict:
                file_logit_dict[filenames[i]] += final_logits[i,:]
            else:
                file_logit_dict[filenames[i]] = final_logits[i,:]
                file_label_dict[filenames[i]] = class_labels[i]

        for key,val in file_logit_dict.items():
            pred = val.argmax(dim=-1)
            results["preds"].append(pred)
            results["labels"].append(file_label_dict[key])

        if (epoch==epochs-1):
            logitname = self.mode+".pt"
            torch.save(final_logits, logitname)
            torch.save(filenames, 'files.pt')
            torch.save(class_labels, 'labels.pt')
            print("***FILES SAVED***")

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )

        average_loss = total_loss / len(self.val_loader)

        pca = compute_pca(
            np.array(results["labels"]), np.array(results["preds"])
        )

        total = 0
        for key, val in pca.items():
            total += val
            print(f"Class {key} accuracy: {val*100:2.2f}")

        pca_avg = total/10
        print(f"Average accuracy: {pca_avg}")

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )


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


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
      f"CNN_"
      f"{args.mode}_"
      f"decay={args.decay}_"
      f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
