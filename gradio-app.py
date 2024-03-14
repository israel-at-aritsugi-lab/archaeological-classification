import os
import pathlib
import shutil
from datetime import datetime

import gradio as gr
import lib.augmentations
import torch
from convigure import Conf
from lib import models
from lib.utils import get_model_filename
from torchvision import transforms


def predict(image):
    if image is None:
        return "Please select an image."

    image = aug(image=image)["image"]
    input = transforms.ToTensor()(image).unsqueeze(0).to(conf.model.device)

    with torch.no_grad():
        pred, _ = model(input)

    pred = pred.cpu()[0]
    pred = torch.nn.functional.softmax(pred, dim=0)
    confidences = {classnames[i]: float(pred[i]) for i in range(conf.model.n_classes)}

    return confidences


def save_image(files, category):
    if files is None:
        return "Please select images."
    elif category == '':
        return "Please select a category."

    upload_dir = pathlib.Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for f in files:
        file = pathlib.Path(f.name)
        dest = upload_dir / f"{now}-({category})-{file.name}"

        shutil.copy2(file, dest)

    file_list = "\r\n".join([f.name.split("/")[-1] for f in files])

    return f"Image(s) uploaded successfully:\r\n\r\n{file_list}"


conf_path = os.getcwd() + "/conf/clahe-2-rotate+flip+transpose+/conf-1.json"
conf = Conf.load_json(conf_path)
model = models.select(conf.model.type, conf.model.n_classes)
model_name = get_model_filename(conf)
model_path = pathlib.Path(conf.model.dir) / model_name
model.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
)

model.to(conf.model.device)
model.eval()

aug = lib.augmentations.compose(conf.datasets.test.aug)
classnames = Conf.load_json("/nas.dbms/randy/datasets/arch/7_classes/classnames.json")

with gr.Blocks() as app:
    gr.Markdown("# Archeology App")

    with gr.Tab("Classifier"):
        with gr.Row():
            with gr.Column():
                classifier_input = gr.Image(shape=(400, 400))
                classifier_button = gr.Button("Classify")

            with gr.Column():
                classifier_output = gr.Label(num_top_classes=7)

    with gr.Tab("Uploader"):
        with gr.Row():
            with gr.Column():
                uploader_file = gr.File(file_count="multiple", file_types=["image"])
                uploader_category = gr.Dropdown(
                    choices=[
                        "Ailanthoides",
                        "Barley",
                        "Maize weevil",
                        "Millet",
                        "Rice",
                        "Perilla",
                        "Wheat",
                    ],
                    label="Image category",
                    info="The uploaded images will be labeled with this category",
                )
                uploader_button = gr.Button("Upload")

            with gr.Column():
                uploader_output = gr.Label()

    classifier_button.click(
        fn=predict, inputs=classifier_input, outputs=classifier_output
    )
    uploader_button.click(
        fn=save_image,
        inputs=[uploader_file, uploader_category],
        outputs=uploader_output,
    )

app.launch(share=False)
