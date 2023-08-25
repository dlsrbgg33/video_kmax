# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
# modified by Qihang Yu
import copy
import logging

import numpy as np
import torch
import random

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances, polygons_to_bitmask

from pycocotools import mask as coco_mask
import pycocotools.mask as mask_util

import os

__all__ = ["InstancekMaXDeepLabDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def build_transform_gen(cfg, is_train, scale_ratio=1.0):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        image_size = cfg.INPUT.IMAGE_SIZE
        # assert is_train

        min_scale = cfg.INPUT.MIN_SCALE * scale_ratio
        max_scale = cfg.INPUT.MAX_SCALE * scale_ratio

        # Augmnetation order majorlly follows deeplab2: resize -> autoaug (color jitter) -> random pad/crop -> flip
        # But we alter it to  resize -> color jitter -> flip -> pad/crop, as random pad is not supported in detectron2.
        # The order of flip and pad/crop does not matter as we are doing random padding/crop anyway.
        augmentation = [
            # Unlike deeplab2 in tf, here the interp will be done in uin8 instead of float32.
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size[0], target_width=image_size[1]
            ), # perofrm on uint8 or float32
            # ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT), # performed on uint8
            # Unlike deeplab2 in tf, here the padding value for image is 128, for label is 255. Besides, padding here will only pad right and bottom.
            # T.FixedSizeCrop(crop_size=(image_size, image_size)),

            # We only perform crop, and do padding manually as the padding value matters. This will crop the image to min(h, image_size).
            T.RandomCrop(crop_type="absolute", crop_size=(image_size[0], image_size[1])),
            T.RandomFlip(),
        ]
    else:
        # Resize
        image_size = cfg.INPUT.IMAGE_SIZE
        augmentation = [
            # Unlike deeplab2 in tf, here the interp will be done in uin8 instead of float32.
            T.ResizeScale(
                min_scale=1, max_scale=1, target_height=image_size[0], target_width=image_size[1]
            ), # perofrm on uint8 or float32
        ]

    return augmentation


def build_color_jitter_transform(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    This is color jittering for sequence-level augmentation.
    Returns:
        list[Augmentation]
    """
    augmentation = [
        ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT), # performed on uint8
    ]
    return augmentation

class InstancekMaXDeepLabDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by kMaX-DeepLab.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        tfm_gens_copy_paste,
        tfm_color_jitter_gens,
        image_format,
        image_size,
        dataset_name,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            tfm_gens_copy_paste: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`
            image_size: expected image size
        """
        self.tfm_gens = tfm_gens
        self.tfm_color_jitter_gens = tfm_color_jitter_gens
        self.tfm_gens_copy_paste = tfm_gens_copy_paste

        if is_train:
            logging.getLogger(__name__).info(
                "[InstancekMaXDeepLabDatasetMapper] Full TransformGens used in training: {}, {} and seq for {}".format(
                    str(self.tfm_gens), str(self.tfm_gens_copy_paste), str(self.tfm_color_jitter_gens)
                )
            )
        else:
            logging.getLogger(__name__).info(
                "[InstancekMaXDeepLabDatasetMapper] Full TransformGens used in testing: {}".format(
                    str(self.tfm_gens)
                )
            )
        self.img_format = image_format
        self.is_train = is_train
        self.image_size = image_size
        self.dataset_name = dataset_name


        dataset_root = './datasets'

        from ..datasets import ytvis
        if dataset_name == 'ytvis_2019':
            image_dir = os.path.join(dataset_root, "ytvis_2019/train/JPEGImages")
            json_file = os.path.join(dataset_root, "ytvis_2019/train.json")
            self.dataset_dict_all = ytvis.load_ytvis_json(
                json_file=json_file, image_root=image_dir, dataset_name='ytvis_2019')

        elif dataset_name == 'ytvis_2021':
            image_dir = os.path.join(dataset_root, "ytvis_2021/train/JPEGImages")
            json_file = os.path.join(dataset_root, "ytvis_2021/train.json")
            self.dataset_dict_all = ytvis.load_ytvis_json(
                json_file=json_file, image_root=image_dir, dataset_name='ytvis_2021')
                
        self.videoname2idx = {}
        for idx, dataset_dict in enumerate(self.dataset_dict_all):
            self.videoname2idx[dataset_dict["video_id"]] = idx


        # implement sampling hyperparameters
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        tfm_gens_copy_paste = build_transform_gen(cfg, is_train, scale_ratio=0.5)
        tfm_color_jitter_gens = build_color_jitter_transform(cfg, is_train)
        
        # config for video
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        dataset_name = '_'.join(cfg.DATASETS.TRAIN[0].split('_')[:2])
        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "tfm_gens_copy_paste": tfm_gens_copy_paste,
            "tfm_color_jitter_gens": tfm_color_jitter_gens,
            "image_format": cfg.INPUT.FORMAT,
            "image_size": cfg.INPUT.IMAGE_SIZE,
            "dataset_name": dataset_name,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
        }
        return ret

    def read_dataset_dict(self, dataset_dict, is_copy_paste=False):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        video_length = dataset_dict["length"]

        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)


        image_for_concat = []
        if self.is_train:
            for frame_idx in selected_idx:
                current_dataset_dict_for_color_jitter = copy.deepcopy(dataset_dict["file_names"][frame_idx])
                image = utils.read_image(current_dataset_dict_for_color_jitter, format=self.img_format)
                utils.check_image_size(dataset_dict, image)
                height, width, _ = image.shape
                image_for_concat.append(image)
            image_concat = np.concatenate(image_for_concat, axis=0)
            image_color_jittered, transfomr_color = T.apply_transform_gens(self.tfm_color_jitter_gens, image_concat)
            color_jittered_images = np.split(image_color_jittered, self.sampling_frame_num, axis=0)


        datast_dict_list = []
        pan_seg_gt_list = []

        for idx, frame_idx in enumerate(selected_idx):
            # make the list for multiple frames
            current_dataset_dict = copy.deepcopy(dataset_dict)
            current_dataset_dict["image"] = []
            current_dataset_dict["is_real_pixels"] = []
            if self.is_train:
                image = color_jittered_images[idx]
            else:
                image = utils.read_image(current_dataset_dict["file_names"][frame_idx], format=self.img_format)
            utils.check_image_size(current_dataset_dict, image)


            if idx == 0:
                if not is_copy_paste:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image, transforms = T.apply_transform_gens(self.tfm_gens_copy_paste, image)
            else:
                image = transforms.apply_image(image)    

            image_shape = image.shape[:2]  # h, w
            current_dataset_dict["image"] = np.ascontiguousarray(image.transpose(2, 0, 1))

            if not self.is_train:
                # If this is for test, we can directly return the unpadded image, as the padding
                # will be handled by size_divisibility
                current_dataset_dict.pop("annotations", None)
                datast_dict_list.append(current_dataset_dict)
                pan_seg_gt_list = None
            else:
                # We pad the image manually, for copy-paste purpose.
                padded_image = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=current_dataset_dict["image"].dtype)
                new_h, new_w = current_dataset_dict["image"].shape[1:]
                offset_h, offset_w = 0, 0 # following the d2 panoptic deeplab implementaiton to only perform bottom/right padding.
                padded_image[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = current_dataset_dict["image"]
                current_dataset_dict["image"] = padded_image

                if "annotations" in current_dataset_dict:

                    _frame_annos = []
                    for anno in current_dataset_dict["annotations"][frame_idx]:
                        _anno = {}
                        for k, v in anno.items():
                            _anno[k] = copy.deepcopy(v)
                        _frame_annos.append(_anno)

                    # USER: Implement additional transformations if you have other types of data
                    annos = [
                        utils.transform_instance_annotations(obj, transforms, image_shape)
                        for obj in _frame_annos
                        if obj.get("iscrowd", 0) == 0
                    ]

                    instances = Instances(image_shape)
                    segms = [obj["segmentation"] for obj in annos]
                    masks = []
                    for segm in segms:
                        if isinstance(segm, list):
                            # polygon
                            masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                        elif isinstance(segm, dict):
                            # COCO RLE
                            masks.append(mask_util.decode(segm))
                        elif isinstance(segm, np.ndarray):
                            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                                segm.ndim
                            )
                            # mask array
                            masks.append(segm)
                        else:
                            raise ValueError(
                                "Cannot convert segmentation of type '{}' to BitMasks!"
                                "Supported types are: polygons as list[list[float] or ndarray],"
                                " COCO-style RLE as a dict, or a binary segmentation mask "
                                " in a 2D numpy array of shape HxW.".format(type(segm))
                            )

                    if len(masks) == 0:
                        # Some image does not have annotation (all ignored)
                        gt_masks = torch.zeros((0, image_shape[0], image_shape[1]))
                    else:
                        gt_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks], dim=0)
                    gt_classes = [int(obj["category_id"]) for obj in annos]
                    gt_classes = torch.tensor(gt_classes, dtype=torch.int64)

                    seg_id = [int(obj["id"]) for obj in annos]
                    seg_id = torch.tensor(seg_id, dtype=torch.int64)


                    # padding
                    padded_gt_masks = torch.zeros((gt_masks.shape[0], self.image_size[0], self.image_size[1]), dtype=gt_masks.dtype)
                    is_real_pixels = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.bool)
                    padded_gt_masks[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = gt_masks
                    is_real_pixels[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = True
                    instances.gt_masks = padded_gt_masks
                    if gt_classes is not None:
                        instances.gt_classes = gt_classes
                    if seg_id is not None:
                        instances.seg_id = seg_id
                        

                    current_dataset_dict["is_real_pixels"] = is_real_pixels
                    datast_dict_list.append(current_dataset_dict)
                    pan_seg_gt_list.append(instances)

        return datast_dict_list, pan_seg_gt_list


    def call_copypaste(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        # Read clip.
        dataset_dict_list, pan_seg_gt_list = self.read_dataset_dict(dataset_dict, is_copy_paste=False)

        dataset_dict = {}

        dataset_dict["image"] = []
        dataset_dict["file_names"] = []
        dataset_dict["video_id"] = []
        dataset_dict["height"] = []
        dataset_dict["width"] = []
        dataset_dict["video_names"] = []

        for dataset_dict_each in dataset_dict_list:
            dataset_dict["file_names"].append(dataset_dict_each["file_names"])
            dataset_dict["video_names"].append(dataset_dict_each["video_names"])
            dataset_dict["video_id"].append(dataset_dict_each["video_id"])
            dataset_dict["height"].append(dataset_dict_each["height"])
            dataset_dict["width"].append(dataset_dict_each["width"])     

        if not self.is_train:
            for dataset_dict_each in dataset_dict_list:
                dataset_dict["image"].append(torch.as_tensor(dataset_dict_each["image"]))
            return dataset_dict


        # Read copy-paste image.
        # It should be sometinng like xxx/xxx/xxx/000000139.jpg, etc. we use the last number as a bias to random number.
        main_video_idx = self.videoname2idx[dataset_dict_list[0]["video_id"]]
        random_video_idx = main_video_idx + random.randint(0, len(self.dataset_dict_all) - 1)
        random_video_idx = random_video_idx % len(self.dataset_dict_all)
        dataset_dict_copy_paste = copy.deepcopy(self.dataset_dict_all[random_video_idx])
        dataset_dict_copy_paste_list, pan_seg_gt_copy_paste_list = self.read_dataset_dict(dataset_dict_copy_paste, is_copy_paste=True)


        dataset_dict["instances"] = []
        dataset_dict["is_real_pixels"] = []
        dataset_dict["valid_pixel_num"] = []

        clip_copy_label_list = []

        rand_prob = random.random()
        for i in range(len(dataset_dict_list)):
            copy_paste_masks = np.zeros((pan_seg_gt_list[i].gt_masks.shape[-2], pan_seg_gt_list[i].gt_masks.shape[-1]))

            # we copy all instances (thing) from copy_paste_image to main_image.
            all_ids = list(range(pan_seg_gt_copy_paste_list[i].gt_masks.shape[0]))
        
        
            random.shuffle(all_ids)
            keep_number = random.randint(0, len(all_ids))
        
            if i == 0:
                for j in range(keep_number):
                    copy_paste_masks[pan_seg_gt_copy_paste_list[i].gt_masks[all_ids[j]] > 0] = 1.0
                    clip_copy_label_list.append(all_ids[j])
            else:
                for label_id in clip_copy_label_list:
                    try:
                        copy_paste_masks[pan_seg_gt_copy_paste_list[i].gt_masks[label_id] > 0] = 1.0
                    except:
                        pass

            # if rand_prob < 1.1:
            #     copy_paste_masks = np.zeros_like(copy_paste_masks)

            dataset_mixed = (dataset_dict_list[i]["image"] * (1.0 - copy_paste_masks).astype(dataset_dict_list[i]["image"].dtype) +
                                    dataset_dict_copy_paste_list[i]["image"] * copy_paste_masks.astype(dataset_dict_list[i]["image"].dtype))
            dataset_dict["image"].append(torch.as_tensor(dataset_mixed))

            dataset_real_pixels_mixed = (dataset_dict_list[i]["is_real_pixels"] * (1.0 - copy_paste_masks).astype(dataset_dict_list[i]["is_real_pixels"].dtype) +
                                    dataset_dict_copy_paste_list[i]["is_real_pixels"] * copy_paste_masks.astype(dataset_dict_list[i]["is_real_pixels"].dtype))
            dataset_dict["is_real_pixels"].append(torch.as_tensor(dataset_real_pixels_mixed))

            # remove all pixels that are overwritten.
            new_gt_masks = pan_seg_gt_list[i].gt_masks.numpy()
            new_gt_masks = np.concatenate([new_gt_masks * (1.0 - copy_paste_masks).astype(new_gt_masks.dtype),
                            pan_seg_gt_copy_paste_list[i].gt_masks.numpy() * copy_paste_masks.astype(new_gt_masks.dtype)], axis=0)
            new_gt_masks = new_gt_masks[:, ::4, ::4]
            new_gt_classes = np.concatenate([pan_seg_gt_list[i].gt_classes.numpy(), pan_seg_gt_copy_paste_list[i].gt_classes.numpy()], axis=0)
            
            copy_seg_id = pan_seg_gt_copy_paste_list[i].seg_id.numpy() + 50
            new_seg_id = np.concatenate([pan_seg_gt_list[i].seg_id.numpy(), copy_seg_id], axis=0)



            # filter empty masks.
            classes = []
            masks = []
            seg_ids = []
            valid_pixel_num = 0
            for i in range(new_gt_masks.shape[0]):
                valid_pixel_num_ = new_gt_masks[i].sum()
                valid_pixel_num += valid_pixel_num_
                if valid_pixel_num_ > 0:
                    classes.append(new_gt_classes[i])
                    masks.append(new_gt_masks[i])
                    seg_ids.append(new_seg_id[i])

            image_shape = new_gt_masks.shape[1:]  # h, w
            instances = Instances(image_shape)
            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)  
            seg_ids = np.array(seg_ids)
            instances.seg_id = torch.tensor(seg_ids, dtype=torch.int64)        

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, new_gt_masks.shape[1], new_gt_masks.shape[2]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                        torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                    )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"].append(instances)
            dataset_dict["valid_pixel_num"].append(valid_pixel_num)
        
        return dataset_dict

    def __call__(self, dataset_dict):
        
        res = self.call_copypaste(dataset_dict)
        while ("instances" in res and res["instances"][0].gt_masks.shape[0] == 0) or ("valid_pixel_num" in res and res["valid_pixel_num"][0] <= 4096):
            # this gt is empty or contains too many void pixels, let's re-generate one.
            main_video_idx = self.videoname2idx[dataset_dict["video_id"]]
            random_video_idx = main_video_idx + random.randint(0, len(self.dataset_dict_all) - 1)
            random_video_idx = random_video_idx % len(self.dataset_dict_all)
            dataset_dict = copy.deepcopy(self.dataset_dict_all[random_video_idx])
            res = self.call_copypaste(dataset_dict)

        return res