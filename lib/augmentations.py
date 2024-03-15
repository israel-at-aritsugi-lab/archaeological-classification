import albumentations as A
from albumentations.augmentations.blur.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.transforms import *
from albumentations.core.transforms_interface import *


def make_list(augs: list) -> list:
    avail_methods = {
        "SmallestMaxSize": A.SmallestMaxSize,
        "CenterCrop": A.CenterCrop,
        "Normalize": Normalize,
        "CLAHE": CLAHE,
        "GaussianBlur": GaussianBlur,
        "Sharpen": A.augmentations.transforms.Sharpen,
        "RandomBrightnessContrast": RandomBrightnessContrast,
        "Equalize": Equalize,
        "HorizontalFlip": HorizontalFlip,
        "VerticalFlip": VerticalFlip,
        "Flip": Flip,
        "NoOp": NoOp,
        "Transpose": Transpose,
        "Rotate": Rotate,
        "RandomRotate90": RandomRotate90,
        "Perspective": Perspective,
        "ElasticTransform": ElasticTransform,
        "GridDistortion": GridDistortion,
        "ShiftScaleRotate": ShiftScaleRotate,
        "Defocus": Defocus
    }

    aug_list = []

    for aug in augs:
        if aug["method"] in avail_methods:
            name = aug["method"]
            params = aug.get("params", None)
            method = (
                avail_methods[name](**params)
                if params is not None
                else avail_methods[name]()
            )

            aug_list.append(method)

    return aug_list

def compose(aug: list) -> A.Compose:
    return A.Compose(make_list(aug))
