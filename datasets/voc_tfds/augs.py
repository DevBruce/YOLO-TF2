import numpy as np
import albumentations as A


__all__ = ['get_transform']


def get_transform(img_height, img_width, input_height, input_width, val=False):
    if val:  # Validation --> Only resize image
        transform = A.Compose(
            [
                A.Resize(input_height, input_width, p=1.),
            ],
            bbox_params=A.BboxParams(
                format='albumentations',
                min_visibility=0,
                label_fields=['class_indices'],
            ),
        )
    else:  # Training --> Augment data
        h_crop_ratio = np.random.uniform(low=0.4, high=1.)
        w_crop_ratio = np.random.uniform(low=0.4, high=1.)
        h_crop = int(img_height * h_crop_ratio)
        w_crop = int(img_width * w_crop_ratio)
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(width=w_crop, height=h_crop, p=0.5),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Resize(input_height, input_width, p=1.),
            ],
            bbox_params=A.BboxParams(
                format='albumentations',
                min_visibility=0.2,
                label_fields=['class_indices'],
            ),
        )
    return transform
