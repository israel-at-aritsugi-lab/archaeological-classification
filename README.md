# ⚱️🏺📜⛏️ Archaeological Classification 🔎🪨🗺️🦖🕵️‍♀️

## Create a Virtual Environment

```shell
python3 -m venv ~/venv/archaeology
source ~/venv/archaeology/bin/activate
```

## Install Packages

```shell
pip install -r requirements.txt
```

## Train

```shell
python3 train.py <config_file>
```

For example:

```shell
python3 train.py conf/clahe-3-blur-9-rotate+flip+transpose+/conf-1.json
```

## Inference

```shell
python3 test.py <config_file>
```

For example:

```shell
python3 test.py conf/clahe-3-blur-9-rotate+flip+transpose+/conf-1.json
```

The results will be saved in MLFlow. To open, run:

```shell
mlflow ui
```

then open [`http://localhost:5000`](http://localhost:5000).
