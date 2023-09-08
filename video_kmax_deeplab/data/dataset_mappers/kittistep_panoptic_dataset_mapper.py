# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
# modified by Inkyu Shin
import copy
import logging

import numpy as np
import torch
import random

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances

from fvcore.transforms.transform import PadTransform

from transformers import AutoTokenizer
import os
import json

from collections import defaultdict
import re

import torch.nn as nn
from collections import OrderedDict

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop



__all__ = ["KITTIPanopticDatasetMapper"]


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
            # T.RandomFlip(),
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


def build_random_flip_transform():
    """
    Create a list of default :class:`Augmentation` from config.
    This is color jittering for sequence-level augmentation.
    Returns:
        list[Augmentation]
    """
    augmentation = [
        T.RandomFlip(), # performed on uint8
    ]
    return augmentation


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name

def kitti_rgb2id(rgb, label_divisor):
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.int32)
    return rgb[:,:,0] * label_divisor + rgb[:,:,1] * 256 + rgb[:,:,2]



# This is specifically designed for the COCO dataset.
class KITTIPanopticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by kMaX-DeepLab.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

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
        is_test=False,
        *,
        tfm_gens,
        tfm_gens_copy_paste,
        tfm_color_jitter_gens,
        tfm_random_flip_gens,
        image_format,
        image_size,
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
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            tfm_gens_copy_paste: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.tfm_gens_copy_paste = tfm_gens_copy_paste
        self.tfm_random_flip_gens = tfm_random_flip_gens
        self.tfm_color_jitter_gens = tfm_color_jitter_gens
        if is_train:
            logging.getLogger(__name__).info(
                "[KITTIPanopticDeepLab2DatasetMapper] Full TransformGens used in training: {}, {}, {} and seq for {}".format(
                    str(self.tfm_gens), str(self.tfm_gens_copy_paste),
                    str(self.tfm_random_flip_gens),
                    str(self.tfm_color_jitter_gens)
                )
            )
        else:
            logging.getLogger(__name__).info(
                "[KITTIPanopticDeepLab2DatasetMapper] Full TransformGens used in testing: {}".format(
                    str(self.tfm_gens)
                )
            )
        self.img_format = image_format
        self.is_train = is_train
        self.is_test = is_test
        self.image_size = image_size

        image_dirt = "./datasets/kitti-step/train/image_02"
        gt_dirt = "./datasets/kitti-step/panoptic_maps_crowd_mapped_rgb_re/train"
        
        from ..datasets import kitti_step
        meta_data = kitti_step._get_kitti_seg_meta()
        
        dataset_dict_t = kitti_step.load_kitti_panoptic_json(
            image_dirt, gt_dirt, meta_data, is_test
        )

        self.label_divisor = meta_data['label_divisor_data']
        self.thing_dataset_id_to_contiguous_id = meta_data['thing_dataset_id_to_contiguous_id']
        

        self.dataset_dict_all = dataset_dict_t 
        self.videoname2idx = {}
        for idx, dataset_dict in enumerate(self.dataset_dict_all):
            self.videoname2idx[dataset_dict[0]["video_id"]] = idx

        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle


    @classmethod
    def from_config(cls, cfg, is_train=True, is_test=False):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        tfm_gens_copy_paste = build_transform_gen(cfg, is_train, scale_ratio=0.5)
        tfm_color_jitter_gens = build_color_jitter_transform(cfg, is_train)
        tfm_random_flip_gens = build_random_flip_transform()

        # config for video
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL
        
        ret = {
            "is_train": is_train,
            "is_test": is_test,
            "tfm_gens": tfm_gens,
            "tfm_gens_copy_paste": tfm_gens_copy_paste,
            "tfm_color_jitter_gens": tfm_color_jitter_gens,
            "tfm_random_flip_gens": tfm_random_flip_gens,
            "image_format": cfg.INPUT.FORMAT,
            "image_size": cfg.INPUT.IMAGE_SIZE,
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
        
        # load clip
        video_length = dataset_dict[0]["length"]
        
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval )
            
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)
        

        datast_dict_list = []
        pan_seg_gt_list = []
        
        image_for_concat = []
        if self.is_train:
            for frame_idx in selected_idx:
                current_dataset_dict_for_color_jitter = copy.deepcopy(dataset_dict[frame_idx])
                image = utils.read_image(current_dataset_dict_for_color_jitter["file_name"], format=self.img_format)
                utils.check_image_size(current_dataset_dict_for_color_jitter, image)
                height, width, _ = image.shape
                image_for_concat.append(image)
            image_concat = np.concatenate(image_for_concat, axis=0)
            # apply random flip
            image_concat, transform_flip = T.apply_transform_gens(self.tfm_random_flip_gens, image_concat)
            image_color_jittered, transfomr_color = T.apply_transform_gens(self.tfm_color_jitter_gens, image_concat)
            color_jittered_images = np.split(image_color_jittered, self.sampling_frame_num, axis=0)
        

        for idx, frame_idx in enumerate(selected_idx):
            # make the list for multiple frames
            current_dataset_dict = copy.deepcopy(dataset_dict[frame_idx])
            current_dataset_dict["image"] = []
            current_dataset_dict["is_real_pixels"] = []
            if self.is_train:
                image = color_jittered_images[idx]
            else:
                image = utils.read_image(current_dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(current_dataset_dict, image)

            if idx == 0:
                if not is_copy_paste:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image, transforms = T.apply_transform_gens(self.tfm_gens_copy_paste, image)
            else:
                image = transforms.apply_image(image)    

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
                #offset_h, offset_w = np.random.randint(0, self.image_size[0] - new_h + 1), np.random.randint(0, self.image_size[1] - new_w + 1)
                offset_h, offset_w = 0, 0 # following the d2 panoptic deeplab implementaiton to only perform bottom/right padding.
                padded_image[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = current_dataset_dict["image"]
                current_dataset_dict["image"] = padded_image
                
                if "pan_seg_file_name" in current_dataset_dict:
                    pan_seg_gt = utils.read_image(current_dataset_dict.pop("pan_seg_file_name"), "RGB")

                    
                    # # apply the same transformation to panoptic segmentation
                    pan_seg_gt = transform_flip.apply_segmentation(pan_seg_gt)
                    pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

                    # pan_seg_gt = kitti_rgb2id(pan_seg_gt, self.label_divisor) # int32 # H x W
                    from panopticapi.utils import rgb2id

                    pan_seg_gt = rgb2id(pan_seg_gt) # int32 # H x W
                    
                    # similarily, we manually pad the label, and we use label -1 to indicate those padded pixels.
                    # In this way, we can masking out the padded pixels values to -1 after normalization, which aligns the
                    # behavior between training and testing.
                    padded_pan_seg_gt = -1 * np.ones((self.image_size[0], self.image_size[1]), dtype=pan_seg_gt.dtype)
                    is_real_pixels = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.bool)
                    padded_pan_seg_gt[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = pan_seg_gt
                    is_real_pixels[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = True
                    current_dataset_dict["is_real_pixels"] = is_real_pixels
                    datast_dict_list.append(current_dataset_dict)
                    pan_seg_gt_list.append(padded_pan_seg_gt)

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
        dataset_dict["image_id"] = []
        dataset_dict["height"] = []
        dataset_dict["width"] = []

        for dataset_dict_each in dataset_dict_list:
            dataset_dict["file_names"].append(dataset_dict_each["file_name"])
            dataset_dict["video_id"].append(dataset_dict_each["video_id"])
            dataset_dict["image_id"].append(dataset_dict_each["image_id"])
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
        dataset_dict["sem_seg_gt"] = []
        dataset_dict["is_real_pixels"] = []
        clip_copy_label_list = []

        seg_id_to_idx = {}
        classes = {}
        masks = {}


        rand_prob = random.random()

        for i in range(len(dataset_dict_list)):
        
            # Copy data_dict_copy_paste onto data_dict. 0 means keep original pixel, 1 means use copy-paste pixel.
            copy_paste_masks = np.zeros((pan_seg_gt_list[i].shape[-2], pan_seg_gt_list[i].shape[-1]))
            segments_info_copy_paste = pan_seg_gt_copy_paste_list[i]
            
            all_ids = np.unique(segments_info_copy_paste)
            thing_ids = []

            for id in all_ids:
                if id in self.thing_dataset_id_to_contiguous_id:
                    thing_ids.append(id)


            # Shuffle and randomly select kept label ids.
            random.shuffle(all_ids)
            keep_number = random.randint(0, len(all_ids))

            if i == 0:
                for index, label_id in enumerate(all_ids):
                    # randomly copy labels, but keep all thing classes.
                    # if index < keep_number or label_id in thing_ids:
                    if index < keep_number:
                        copy_paste_masks[pan_seg_gt_copy_paste_list[i] == label_id] = 1
                        clip_copy_label_list.append(label_id)
            else:
                for label_id in clip_copy_label_list:
                    copy_paste_masks[pan_seg_gt_copy_paste_list[i] == label_id] = 1

            # random probability to copy-paste. (probability is 0.50)
            # if rand_prob < 1.1:
            #     copy_paste_masks = np.zeros_like(copy_paste_masks)

            # We merge the image and copy-paste image based on the copy-paste mask.
            # 3 x H x W
            dataset_mixed = (dataset_dict_list[i]["image"] * (1.0 - copy_paste_masks).astype(dataset_dict_list[i]["image"].dtype) +
                                    dataset_dict_copy_paste_list[i]["image"] * copy_paste_masks.astype(dataset_dict_list[i]["image"].dtype))
            dataset_dict["image"].append(torch.as_tensor(dataset_mixed))

            dataset_real_pixels_mixed = (dataset_dict_list[i]["is_real_pixels"] * (1.0 - copy_paste_masks).astype(dataset_dict_list[i]["is_real_pixels"].dtype) +
                                    dataset_dict_copy_paste_list[i]["is_real_pixels"] * copy_paste_masks.astype(dataset_dict_list[i]["is_real_pixels"].dtype))
            dataset_dict["is_real_pixels"].append(torch.as_tensor(dataset_real_pixels_mixed))
            
            # We set all ids in copy-paste image to be negative, so that there will be no overlap between original id and copy-paste id.
            pan_seg_gt_copy_paste = -pan_seg_gt_copy_paste_list[i]
            pan_seg_gt = pan_seg_gt_list[i]
            pan_seg_gt = (pan_seg_gt * (1.0 - copy_paste_masks).astype(pan_seg_gt.dtype) +
                        pan_seg_gt_copy_paste * copy_paste_masks.astype(pan_seg_gt.dtype))

            # We use 4x downsampled gt for final supervision.
            pan_seg_gt = pan_seg_gt[::4, ::4]

            sem_seg_gt = -np.ones_like(pan_seg_gt) # H x W, init with -1

            # We then process the obtained pan_seg_gt to training format.
            image_shape = dataset_dict_list[i]["image"].shape[1:]  # h, w
            
            instances = Instances(image_shape)
            classes = []
            masks = []
            seg_id_list = []


            stuff_class_to_idx = {}

            segments_info = pan_seg_gt_list[i]
            segments_info_ids = np.unique(segments_info)
            
            for seg_id in segments_info_ids:
                class_id = seg_id // self.label_divisor
                if (class_id != 255) and (class_id != -1):
                    binary_mask = (pan_seg_gt == seg_id)
                

                    if np.any(binary_mask):
                        sem_seg_gt[binary_mask] = class_id
                        if class_id not in self.thing_dataset_id_to_contiguous_id:
                            if class_id in stuff_class_to_idx:
                                raise ValueError('class_id should not already be in stuff_class_to_idx!')
                            else:
                                stuff_class_to_idx[class_id] = len(masks)
                        classes.append(class_id)
                        masks.append(binary_mask)
                        seg_id_list.append(seg_id)


            segments_info_copy = pan_seg_gt_copy_paste_list[i]
            segments_info_copy_ids = np.unique(segments_info_copy)


            for seg_id in segments_info_copy_ids:
                class_id = seg_id // self.label_divisor
                if (class_id != 255) and (class_id != -1):
                    binary_mask = (pan_seg_gt == -seg_id)
                    
                    if np.any(binary_mask):
                        sem_seg_gt[binary_mask] = class_id
                        if class_id not in self.thing_dataset_id_to_contiguous_id:
                            if class_id in stuff_class_to_idx:
                                # Merge into original stuff masks. 
                                masks[stuff_class_to_idx[class_id]] = np.logical_or(masks[stuff_class_to_idx[class_id]], binary_mask)
                                continue
                            else:
                                stuff_class_to_idx[class_id] = len(masks)

                        classes.append(class_id)
                        masks.append(binary_mask)
                        seg_id_list.append(seg_id)

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            seg_id = np.array(seg_id_list)
            instances.seg_id = torch.tensor(seg_id, dtype=torch.int64)
            sem_seg_gt = torch.tensor(sem_seg_gt, dtype=torch.int64)
        
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"].append(instances)
            dataset_dict["sem_seg_gt"].append(sem_seg_gt)

        return dataset_dict


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """        
        res = self.call_copypaste(dataset_dict)
            
        while "instances" in res and res["instances"][0].gt_masks.shape[0] == 0:
            # this gt is empty, let's re-generate one.
            main_video_idx = self.videoname2idx[dataset_dict[0]["video_id"]]
            random_video_idx = main_video_idx + random.randint(0, len(self.dataset_dict_all) - 1)
            random_video_idx = random_video_idx % len(self.dataset_dict_all)
            dataset_dict = copy.deepcopy(self.dataset_dict_all[random_video_idx])
            res = self.call_copypaste(dataset_dict)
        return res
