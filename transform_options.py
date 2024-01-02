def get_target_transformations(target_height, target_width):
    
    import albumentations as A
    
    return list([
        # All Augmentations with probable flip
        ## Noisy Augmentation:
        A.Compose(
        [
            A.RandomCrop(width=450, height=450, p=.4),
            A.RandomRotate90(p=.3),
            A.GaussNoise(var_limit=(10., 40.)),
            A.HorizontalFlip(p=.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
        A.Compose([
            A.RandomCrop(width=450, height=450, p=.4),
            A.RandomRotate90(p=.3),
            A.ISONoise(color_shift=(0.01, 0.10)),
            A.HorizontalFlip(p=.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
        A.Compose([
            A.RandomCrop(width=450, height=450, p=.4),
            A.RandomRotate90(p=.3),
            A.MultiplicativeNoise(multiplier=(1, 1.5)),
            A.HorizontalFlip(p=.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
        # Blurred Augmentation
        A.Compose([
            A.RandomCrop(width=450, height=450, p=.4),
            A.RandomRotate90(p=.2),
            A.GaussianBlur(blur_limit=(3,5)),
            A.HorizontalFlip(p=.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
        A.Compose([
            A.RandomCrop(width=450, height=450, p=.4),
            A.RandomRotate90(p=.2),
            A.GaussianBlur(blur_limit=(3,5)),
            A.HorizontalFlip(p=.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
        A.Compose([
            A.RandomCrop(width=450, height=450, p=.4),
            A.RandomRotate90(p=.2),
            A.AdvancedBlur(blur_limit=(7, 11)),
            A.HorizontalFlip(p=.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
        # Distortion Augmentation
        A.Compose([
            A.RandomCrop(width=450, height=450, p=.4),
            A.OpticalDistortion(distort_limit=(-0.05, 0.05)),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
        A.Compose([
            A.RandomCrop(width=450, height=450, p=.4),
            A.GridDistortion(num_steps=5, normalized=True),
            A.Resize(target_height,target_width, always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])),
    ])