
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
import random

# # Load vipseg categories with json file
# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# json_path = os.path.join(_root, 'burst/coco_annotations/ori_categories.json')
# with open(json_path, 'r') as f:
#     data = json.load(f)
# BURST_CATEGORIES = data

def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

def _get_burst_seg_meta():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    # thing_classes = [k["name"] for k in VIPSEG_CATEGORIES if k["isthing"] == 1]
    # thing_colors = [k["color"] for k in VIPSEG_CATEGORIES if k["isthing"] == 1]
    thing_classes = [k["name"] for k in BURST_CATEGORIES]
    # want to have the random colors for number of thing classes
    thing_colors = generate_colors(len(thing_classes))
    stuff_classes = [k["name"] for k in BURST_CATEGORIES]
    stuff_colors = thing_colors

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

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

    for i, cat in enumerate(BURST_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_burst_video_annos_sem_seg(
    name, metadata, image_root, panoptic_root, panoptic_json
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
        lambda: load_burst_video_json(panoptic_json, image_root, panoptic_root, metadata, is_test),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        evaluator_type="video_panoptic_seg",
        ignore_label=255,
        label_divisor=10000,
        **metadata,
    )
    

# Video Loader
def load_burst_video_json(json_file, video_dir, gt_dir, meta, is_test=False):
    """
    Args:
        video_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    
    dataset_dicts = []
    
    if is_test:
        for video in json_info["videos"]:
            video_id = video["video_id"]
            video_length = len(video["images"])
            video_path = os.path.join(video_dir, video_id)
            ret = []
            for idx, image in enumerate(video["images"]):
                image_file = os.path.join(video_path, os.path.splitext(image["file_name"])[0] + ".jpg")
                ret.append({
                    "file_name": image_file,
                    "image_id": image["id"],
                    "length": video_length,
                    "video_id": video_id
                })
            dataset_dicts.append(ret)
        return dataset_dicts
    else:
        for video_ann in json_info["annotations"]:
            # len(video_ann) is the number of video folders
            ret = []
            video_id = video_ann["video_id"]
            video_length = len(video_ann["annotations"])
            for idx, img_ann in enumerate(video_ann["annotations"]):
                
                image_id = img_ann["image_id"]
                
                video_path = os.path.join(video_dir, video_id)
                gt_path = os.path.join(gt_dir, video_id)
                # TODO: currently we assume image and label has the same filename but
                # different extension, and images have extension ".jpg" for COCO. Need
                # to make image extension a user-provided argument if we extend this
                # function to support other COCO-like datasets.

                image_file = os.path.join(video_path, os.path.splitext(img_ann["file_name"])[0] + ".jpg")
                label_file = os.path.join(gt_path, img_ann["file_name"].replace('jpg', 'png'))
 
                segments_info = [_convert_category_id(x, meta) for x in img_ann["segments_info"]]
                ret.append(
                    {
                        "file_name": image_file,
                        "image_id": image_id,
                        "pan_seg_file_name": label_file,
                        "segments_info": segments_info,
                        "length": video_length,
                        "video_id": video_id
                    }
                )
            assert len(ret), f"No images found in {video_path}!"
            assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
            assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
        
            dataset_dicts.append(ret)
        
        return dataset_dicts
        
