
# Reference: https://github.com/MasterBin-IIAU/UNINEXT/blob/master/projects/UNINEXT/uninext/data/datasets/builtin.py

# Registry for multiple dataset is implemented
# - COCO (panoptic)
# - VIPSeg (panoptic)


import os

from detectron2.data import MetadataCatalog

from .coco import (
    register_coco_panoptic_annos_sem_seg,
    _get_coco_meta
)

# ==== Predefined splits for COCO Panoptic datasets ===========
_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
}

def register_all_coco_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        register_coco_panoptic_annos_sem_seg(
            prefix,
            _get_coco_meta(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        
from .vip_seg import (
    register_vipseg_panoptic_annos_sem_seg,
    _get_vip_seg_meta
)

# ==== Predefined splits for VIPSeg VIDEO Panoptic datasets ===========
_PREDEFINED_SPLITS_VIPSEG_VIDEO_PANOPTIC = {
    "vipseg_train_video_panoptic": (
        # This is the original panoptic annotation directory
        "vip_seg/train_images",
        # converted for coco format of VIPSeg
        # reference from https://github.com/VIPSeg-Dataset/VIPSeg-Dataset
        "vip_seg/panoptic_gt_VIPSeg_train.json",
        "vip_seg/panomasksRGB",
    ),
    # "vipseg_val_video_panoptic": (
    #     "vip_seg/val_images",
    #     "vip_seg/panoptic_gt_VIPSeg_val.json",
    #     "vip_seg/panomasksRGB",
    # ),
    "vipseg_val_video_panoptic": (
        "vip_seg/car-turn",
        "vip_seg/car_turn.json",
        "vip_seg/panomasksRGB",
    ),
    "vipseg_test_video_panoptic": (
        "vip_seg/test_images_720p",
        "vip_seg/panoptic_gt_VIPSeg_test_rev.json",
        "vip_seg/panomasksRGB",
    ),
    "burst_pseudo_train_video_panoptic": (
        "burst/frames/train",
        "burst/pseudo_anno/train.json",
        "burst/panopRGB_pseudo"
    ),
}
def register_all_vipseg_video_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (image_root, panoptic_json, panoptic_root),
    ) in _PREDEFINED_SPLITS_VIPSEG_VIDEO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_video_panoptic")]

        register_vipseg_panoptic_annos_sem_seg(
            prefix,
            _get_vip_seg_meta(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json)
        )



from .kitti_step import (
    register_kitti_panoptic_annos_sem_seg,
    _get_kitti_seg_meta
)

# ==== Predefined splits for VIPSeg VIDEO Panoptic datasets ===========
_PREDEFINED_SPLITS_KITTI_VIDEO_PANOPTIC = {
    "kitti_train_video_panoptic": (
        # This is the original panoptic annotation directory
        "kitti-step/train/image_02",
        # "kitti-step/panoptic_gt_KITTI_val.json",
        "kitti-step/panoptic_gt_KITTI_val.json",
        # "kitti-step/panoptic_maps_crowd_mapped_rgb/train",
        "kitti-step/panoptic_maps_crowd_mapped_rgb_re/train",
    ),
    "kitti_val_video_panoptic": (
        "kitti-step/val",
        # "kitti-step/panoptic_gt_KITTI_val_nocrowded_re.json",
        "kitti-step/panoptic_gt_KITTI_val_crowded.json",
        # "kitti-step/panoptic_maps_nocrowd_re/val",
        "kitti-step/panoptic_maps_crowd_mapped_rgb_re/val",
    ),
}
def register_all_kitti_video_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (image_root, panoptic_json, panoptic_root),
    ) in _PREDEFINED_SPLITS_KITTI_VIDEO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_video_panoptic")]

        register_kitti_panoptic_annos_sem_seg(
            prefix,
            _get_kitti_seg_meta(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, panoptic_root)
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    
    # COCO
    register_all_coco_panoptic_annos_sem_seg(_root)
    register_all_vipseg_video_panoptic_annos_sem_seg(_root)
    # register_all_burst_video_annos_sem_seg(_root)
    register_all_kitti_video_panoptic_annos_sem_seg(_root)
    
    