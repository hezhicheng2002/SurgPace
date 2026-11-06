import os
from PIL import Image, ImageFile
import torch
import ast
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except Exception:
    BICUBIC = Image.BICUBIC
from ..utils.data_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagePreprocess:
    def __init__(self, image_processor, data_args={}):
        self.image_aspect_ratio = getattr(data_args, "image_aspect_ratio", None)
        self.image_processor = image_processor
        self.image_grid_pinpoints = getattr(data_args, "image_grid_pinpoints", None)

    def __call__(self, image):
        if self.image_aspect_ratio == "pad":
            image = self.expand2square(
                image, tuple((int(x * 255) for x in self.image_processor.image_mean))
            )
        elif self.image_aspect_ratio == "anyres":
            image = self.process_anyres_image(
                image, self.image_processor, self.image_grid_pinpoints
            )
            return image
        image = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        return image

    @classmethod
    def expand2square(cls, pil_img, background_color):
        (width, height) = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    @classmethod
    def process_anyres_image(cls, image, processor, grid_pinpoints):
        if type(grid_pinpoints) is list:
            possible_resolutions = grid_pinpoints
        else:
            possible_resolutions = ast.literal_eval(grid_pinpoints)
        best_resolution = select_best_resolution(image.size, possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)
        patches = divide_to_patches(image_padded, processor.crop_size["height"])
        image_original_resize = image.resize(
            (processor.size["shortest_edge"], processor.size["shortest_edge"])
        )
        image_patches = [image_original_resize] + patches
        image_patches = [
            processor(image_patch, return_tensors="pt")["pixel_values"][0]
            for image_patch in image_patches
        ]
        return torch.stack(image_patches, dim=0)


class TeacherImagePreprocess:
    def __init__(self, size=1024, transform_type="padded_resize"):
        self.size = int(size)
        self.transform_type = str(transform_type or "padded_resize")
        self.clip_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_std = [0.26862954, 0.26130258, 0.27577711]

    def __call__(self, image):
        if self.transform_type == "padded_resize":
            image = ImagePreprocess.expand2square(
                image, tuple((int(x * 255) for x in self.clip_mean))
            )
            try:
                resample = Image.Resampling.BICUBIC
            except AttributeError:
                resample = Image.BICUBIC
            image = image.resize((self.size, self.size), resample=resample)
        elif self.transform_type == "resized_center_crop":
            image = transforms.Resize(self.size, interpolation=BICUBIC)(image)
            image = transforms.CenterCrop(self.size)(image)
        elif self.transform_type == "random_resized_crop":
            image = transforms.RandomResizedCrop(
                size=(self.size, self.size),
                scale=(0.9, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=BICUBIC,
            )(image)
        else:
            raise ValueError(f"unknown teacher image transform: {self.transform_type}")
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=self.clip_mean, std=self.clip_std)(image)
        return image
