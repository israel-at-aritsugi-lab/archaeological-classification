import os
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")


def save_image(files, category):
    if files is None:
        return "Please select images."
    elif category is None:
        return "Please select a category."

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for f in files:
        file = Path(f.name)
        dest = upload_dir / f"{now}-({category})-{file.name}"

        shutil.copy2(file, dest)

    file_list = "\r\n".join([f.name.split("/")[-1] for f in files])

    return f"Image(s) uploaded successfully:\r\n\r\n{file_list}"


gr.Interface(
    title="Archeology Image Uploader App.",
    fn=save_image,
    inputs=[
        gr.File(file_count="multiple", file_types=["image"]),
        gr.Dropdown(
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
        ),
    ],
    outputs="text",
    allow_flagging="never",
).launch(share=False, auth=(username, password))
