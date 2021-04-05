from .ssd import get_coco_for_ssd
from deeplite_torch_zoo.src.objectdetection.configs.coco_config import MISSING_IDS, DATA
from deeplite_torch_zoo.src.objectdetection.mb_ssd.config.mobilenetv1_ssd_config import (
    MOBILENET_CONFIG,
)

__all__ = ["get_coco_for_mb2_ssd", "get_coco_gm_for_mb2_ssd"]


def _get_coco_for_mb2_ssd(
    data_root,
    batch_size=32,
    train_ann_file=None,
    train_dir=None,
    val_ann_file=None,
    val_dir=None,
    missing_ids=[],
    classes=[],
    **kwargs
):
    return get_coco_for_ssd(
        data_root=data_root,
        config=MOBILENET_CONFIG(),
        batch_size=batch_size,
        train_ann_file=train_ann_file,
        train_dir=train_dir,
        val_ann_file=val_ann_file,
        val_dir=val_dir,
        missing_ids=missing_ids,
        classes=classes,
    )


def get_coco_gm_for_mb2_ssd(
    data_root,
    batch_size=32,
    **kwargs
):
    return _get_coco_for_mb2_ssd(
            data_root=data_root,
            batch_size=batch_size,
            train_ann_file="train_data_COCO.json",
            train_dir="images/train",
            val_ann_file="test_data_COCO.json",
            val_dir="images/test",
            classes=["class1", "class2", "class3", "class4", "class5", "class6"],
        )

def get_coco_for_mb2_ssd(
    data_root,
    batch_size=32,
    **kwargs,
):
    return _get_coco_for_mb2_ssd(
            data_root=data_root,
            batch_size=batch_size,
            train_ann_file="annotations/instances_train2017.json",
            train_dir="train2017",
            val_ann_file="annotations/instances_val2017.json",
            val_dir="val2017",
            classes=DATA["CLASSES"],
            missing_ids=MISSING_IDS,
        )
