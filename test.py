#!/usr/bin/env python3

import pathlib
from os.path import basename

import click
import lib.augmentations
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from convigure import Conf
from lib import engine, models
from lib.dataset import Dataset
from lib.utils import get_model_filename
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import Accuracy, F1Score


def make_prediction_csv(true, pred, labels, df, file_path):
    wrong_idx = np.flatnonzero(pred != true)
    class_list = df.loc[:, "class"].values
    filenames = df.path.values
    wrongs = [[i, filenames[i], class_list[i], labels[pred[i]]] for i in wrong_idx]
    cols = "idx", "filename", "true", "pred"

    file_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(wrongs, columns=cols).to_csv(file_path, index=False)


def make_confusion_matrix(true, pred, labels, title, file_path):
    cm = ConfusionMatrixDisplay.from_predictions(true, pred, display_labels=labels)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    cm.ax_.set_title(title)
    plt.savefig(file_path)


def test(app_conf, model_conf, split: str) -> list:
    dataset_path = pathlib.Path(app_conf.dataset_path)
    data_split_path = dataset_path / split
    csv_path = dataset_path / (split + ".csv")
    df = pd.read_csv(csv_path, engine="pyarrow")
    test_images = df.path.values.tolist()

    model = models.select(model_conf.model.type, model_conf.model.n_classes)
    model_name = get_model_filename(model_conf)
    model_path = pathlib.Path(model_conf.model.dir) / model_name

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=lambda storage, loc: storage.cuda()
            if torch.cuda.is_available()
            else "cpu",
        )
    )

    model.to(model_conf.model.device)

    test_targets = np.zeros(len(test_images), dtype=np.int_)
    aug = lib.augmentations.compose(model_conf.datasets.test.aug)

    dataset = Dataset(
        data_split_path,
        test_images,
        test_targets,
        aug=aug,
        channel_first=True,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=model_conf.DataLoader.batch_size,
        num_workers=model_conf.DataLoader.num_workers,
        shuffle=False,
    )

    true = df.class_num.values
    pred = engine.predict(model, data_loader, model_conf.model.device)
    pred = np.vstack(pred)
    pred = [np.argmax(p) for p in pred]
    class_unique = Conf.load_json(dataset_path / "classnames.json")

    if app_conf.test.prediction.enabled:
        csv_file_path = (
            pathlib.Path(app_conf.test.prediction.path)
            / model_conf.model.id
            / basename(csv_path)
        )

        make_prediction_csv(true, pred, class_unique, df, csv_file_path)

    if app_conf.test.confusion_matrix.enabled:
        confusion_matrix_file = (
            pathlib.Path(app_conf.test.confusion_matrix.path)
            / model_conf.model.id
            / split
        )

        make_confusion_matrix(
            true, pred, class_unique, model_conf.model.id, confusion_matrix_file
        )

    accuracy_metric = Accuracy(
        task="multiclass", num_classes=model_conf.model.n_classes
    )

    accuracy = float(
        accuracy_metric(
            torch.tensor(pred),
            torch.tensor(true),
        )
    )

    f1_metric = F1Score(
        task="multiclass",
        num_classes=model_conf.model.n_classes,
        average="weighted",
    )

    f1 = float(
        f1_metric(
            torch.tensor(pred),
            torch.tensor(true),
        )
    )

    return accuracy, f1


def test_recursive(app_conf, file):
    if file.is_file():
        print(f"Loading configuration: {file}")

        model_conf = Conf.load_json(file)
        aug_to_log = list(
            filter(
                lambda x: x["method"]
                not in ["SmallestMaxSize", "CenterCrop", "Normalize"],
                model_conf.datasets.train.aug,
            )
        )

        scores = {}

        for split in "test", "real":
            accuracy, f1 = test(
                app_conf,
                model_conf,
                split,
            )

            scores[split] = {"accuracy": accuracy, "f1": f1}

        if app_conf.test.mlflow.enabled:
            mlflow.set_experiment(model_conf.experiment_name)

            for split in "test", "real":
                with mlflow.start_run(run_name=model_conf.model.id) as run:
                    mlflow.log_param("model_type", model_conf.model.type)
                    mlflow.log_param("data", split)
                    mlflow.log_param("fold", model_conf.datasets.train.val_fold)
                    mlflow.log_param("lr", model_conf.optimizer.learning_rate)

                    logged_params = []

                    for method in aug_to_log:
                        if "params" not in method:
                            continue

                        params = dict(
                            filter(
                                lambda k: k[0] != "p" and k[0] not in logged_params,
                                method["params"].items(),
                            )
                        )

                        mlflow.log_params(params)
                        logged_params.extend(params.keys())

                    mlflow.log_metric("accuracy", scores[split]['accuracy'])
                    mlflow.log_metric("f1", scores[split]['f1'])

        print(scores)
    elif file.is_dir():
        for f in file.iterdir():
            test_recursive(app_conf, f)


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

    app_conf = Conf.load_json("project.json")

    for conf_file in conf_files:
        file = pathlib.Path(conf_file)

        if not file.exists():
            print("Config file/directory not found:", file.name)
            continue

        test_recursive(app_conf, file)


if __name__ == "__main__":
    main()
