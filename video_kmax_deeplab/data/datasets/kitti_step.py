
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
from detectron2.data import detection_utils as utils
import numpy as np

# Load vipseg categories with json file
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
json_path = os.path.join(_root, 'kitti-step/panoKITTISTEP_categories.json')
with open(json_path, 'r') as f:
    data = json.load(f)
KITTI_CATEGORIES = data

def _get_kitti_seg_meta():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    # thing_colors = [k["color"] for k in VIPSEG_CATEGORIES if k["isthing"] == 1]
    # import pdb; pdb.set_trace()
    thing_classes = [k["name"] for k in KITTI_CATEGORIES]
    thing_colors = [k["color"] for k in KITTI_CATEGORIES]
    stuff_classes = [k["name"] for k in KITTI_CATEGORIES]
    stuff_colors = [k["color"] for k in KITTI_CATEGORIES]
    # stuff_classes = [k["name"] for k in VIPSEG_CATEGORIES if k["isthing"] == 0]
    # stuff_colors = [k["color"] for k in VIPSEG_CATEGORIES if k["isthing"] == 0]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors
    # meta["stuff_classes"] = thing_classes + stuff_classes
    # meta["stuff_colors"] = thing_colors + stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id_nothing = {}

    # for i in range(len(thing_classes)):
    #     thing_dataset_id_to_contiguous_id[thing_classes[i]] = i
    thing_idx = 0
    for i, cat in enumerate(KITTI_CATEGORIES):
        # if cat["isthing"]:
        #     thing_dataset_id_to_contiguous_id[cat["id"]] = thing_idx
        #     thing_idx += 1
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    # stuff_idx = len(thing_classes)
    # for i, cat in enumerate(VIPSEG_CATEGORIES):
    #     if not cat["isthing"]:
    #         stuff_dataset_id_to_contiguous_id_nothing[cat["id"]] = stuff_idx
    #         stuff_idx += 1


    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    # meta["stuff_dataset_id_to_contiguous_id_nothing"] = stuff_dataset_id_to_contiguous_id_nothing
    meta["label_divisor_data"] = 10000

    return meta


def register_kitti_panoptic_annos_sem_seg(
    name, metadata, image_root, panoptic_json, panoptic_root
):

    panoptic_name = name
    is_test = 'test' in name
    MetadataCatalog.get(panoptic_name).set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        # thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    )
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_kitti_panoptic_json(image_root, panoptic_root, metadata, is_test),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        panoptic_json=panoptic_json,
        image_root=image_root,
        evaluator_type="video_panoptic_seg",
        ignore_label=255,
        label_divisor=10000,
        **metadata,
    )
    

# Video Loader
def load_kitti_panoptic_json(video_dir, gt_dir, meta, is_test=False):
    """
    Args:
        video_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
        
    # load the folder
    dataset_dicts = []

    for subfolder_name in sorted(os.listdir(video_dir)):
        ret = []
        video_id = subfolder_name
        subfolder_path = os.path.join(video_dir, subfolder_name)
        files_in_subfolder = os.listdir(subfolder_path)
        
        video_length = len(files_in_subfolder)
        
        for file in sorted(files_in_subfolder):
            image_id = file.split('.')[0]
            image_file = os.path.join(subfolder_path, file)
            label_file = os.path.join(gt_dir, subfolder_name, file)
            nocrowd_gt = gt_dir.replace('panoptic_maps_crowd_mapped', 'panoptic_maps')
            nocrowd_label_file = os.path.join(nocrowd_gt, subfolder_name, file)
            
            label_image = utils.read_image(label_file, "RGB")
            # nocrowd_label_image = utils.read_image(nocrowd_label_file, "RGB")
            # if 13 in np.unique(label_image[:,:,0]):
            height, width, _ = label_image.shape
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "pan_seg_file_name": label_file,
                    "length": video_length,
                    "video_id": video_id,
                    "height": height,
                    "width": width
                }
            )
            
        assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
        assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    
        dataset_dicts.append(ret)
    return dataset_dicts
        
