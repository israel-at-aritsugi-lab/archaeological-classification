#!/usr/bin/env python3

import os
import pathlib
import platform
import socket
from os.path import exists

import click
import lib.augmentations
import numpy as np
import pandas as pd
import torch
from convigure import Conf
from lib import engine, models
from lib.augduplicateddataset import AugDuplicatedDataset
from lib.es import EarlyStopping
from lib.telegram import Telegram
from lib.utils import (
    Dict,
    avoid_overwriting,
    count_files,
    get_model_filename,
    get_targets,
)
from timer_py import Timer
from torchmetrics import Accuracy, F1Score


def train(dataset_path: pathlib.Path, conf: Dict):
    csv_path = dataset_path / "train.csv"
    df = pd.read_csv(csv_path, engine="pyarrow")
    df_train = df[df.kfold != conf.datasets.train.val_fold].reset_index(drop=True)
    df_val = df[df.kfold == conf.datasets.train.val_fold].reset_index(drop=True)
    duplicate_aug = lib.augmentations.make_list(conf.datasets.train.duplicate_aug)
    aug = lib.augmentations.compose(conf.datasets.train.aug)

    print("Probability augmentation:", aug)
    print("Duplicating augmentation:", duplicate_aug)

    train_dataset = AugDuplicatedDataset(
        dataset_path / "train",
        df_train.path.values,
        df_train.class_num.values,
        duplicate_aug=duplicate_aug,
        aug=aug,
        channel_first=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.DataLoader.batch_size,
        num_workers=conf.DataLoader.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    val_dataset = AugDuplicatedDataset(
        dataset_path / "train",
        df_val.path.values,
        df_val.class_num.values,
        duplicate_aug=duplicate_aug,
        aug=aug,
        channel_first=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=conf.DataLoader.batch_size,
        num_workers=conf.DataLoader.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print("Original train size:", len(df_train))
    print("Augmented train size:", len(train_dataset))

    model = models.select(conf.model.type, conf.model.n_classes)
    model = model.to(device=conf.model.device)
    model_name = get_model_filename(conf)
    model_path = pathlib.Path(conf.model.dir) / model_name
    model_path = avoid_overwriting(model_path)

    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.optimizer.learning_rate,
        betas=conf.optimizer.betas,
        eps=conf.optimizer.epsilon,
        weight_decay=conf.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=conf.scheduler.patience, mode=conf.scheduler.mode
    )

    es = EarlyStopping(patience=conf.scheduler.patience, mode=conf.scheduler.mode)
    accuracy_metric = Accuracy(task="multiclass", num_classes=conf.model.n_classes)

    timer = Timer()
    timer.set_tag(f"{conf.model.type} {conf.model.id}")
    timer.start()

    for epoch in range(conf.training.epoch):
        training_loss = engine.train_fn(
            model, train_loader, optimizer, conf.model.device
        )
        preds, val_loss = engine.evaluate(model, val_loader, conf.model.device)

        print("Tra. loss:", training_loss)
        print("Val. loss:", val_loss)

        pred_list = []

        for vp in preds:
            pred_list.extend(vp)

        preds = [torch.argmax(p) for p in pred_list]
        preds = np.vstack(preds).ravel()

        accuracy = float(
            accuracy_metric(
                torch.tensor(preds),
                # torch.tensor(val_targets),
                get_targets(val_loader),
            )
        )

        scheduler.step(accuracy)
        es(accuracy, model, model_path)

        if es.early_stop:
            print("Early stop")
            break

        print(f"Model: {conf.model.type}, epoch: {epoch}, acc: {accuracy}")

    training_time = timer.stop()

    return accuracy, training_time


def train_file(file, file_iter: int, n_files: int) -> int:
    if file.is_file():
        hostname = socket.gethostname()
        os = platform.system()
        app_conf = Conf.load_json("project.json")
        model_conf = Conf.load_json(file)
        dataset_path = pathlib.Path(app_conf.dataset_path)
        telegram = Telegram(app_conf)
        telegram_msg = f"Machine: {hostname} ({os})\nModel ID: {model_conf.model.id}\nConf: {file.name}"

        print(f"Loading configuration {file}")

        if app_conf.telegram.enabled:
            message = f"Training {file_iter}/{n_files} started 🏃‍♂️\n\n{telegram_msg}"

            telegram.notify(message)

        accuracy, training_time = train(dataset_path, model_conf)

        if app_conf.telegram.enabled:
            message = f"Training {file_iter}/{n_files} finished 🏁\n\n{telegram_msg}\nAccuracy: {accuracy*100}%\nTime: {training_time}"

            telegram.notify(message)

        return file_iter + 1
    elif file.is_dir():
        file_count = file_iter

        for f in file.iterdir():
            file_count = train_file(f, file_count, n_files)

        return file_count


@click.command()
@click.argument(
    "conf-files",
    nargs=-1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
def main(conf_files):
    if conf_files is None:
        print("Configuration file(s) or directory required.")
        exit(1)

    n_files = sum(count_files(f) for f in conf_files)
    file_iter = 1

    for conf_file in conf_files:
        file = pathlib.Path(conf_file)

        if not file.exists():
            print("Config file/directory not found:", file.resolve().absolute())
            continue

        file_iter = train_file(file, file_iter, n_files)


if __name__ == "__main__":
    main()
