import numpy as np
import copy
import cv2
import torch.distributed as dist
import sys
import time

from .misc import NestedTensor, nested_tensor_from_tensor_list
import os
import json

from detectron2.data import MetadataCatalog
from .video_panoptic_visualizer import ColorMode, Visualizer
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# Load vipseg categories with json file
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
json_path = os.path.join(_root, 'vip_seg/panoVIPSeg_categories.json')
with open(json_path, 'r') as f:
    data = json.load(f)
VIPSEG_CATEGORIES = data


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def debug(samples, gt_targets):
    # import pdb
    # pdb.set_trace()
    # if not isinstance(samples, NestedTensor):
    #     samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
    debug_data(samples, gt_targets)

def debug_data(samples, gt_targets):
    import numpy as np
    import copy
    import cv2
    # import torch.distributed as dist
    import sys
    import time
    unorm = UnNormalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
    metadata = MetadataCatalog.get("vipseg_val_video_panoptic")
    # metadata = MetadataCatalog.get("coco_2017_train")
    mean = np.array([127.5, 127.5, 127.5])
    std = np.array([127.5, 127.5, 127.5])
    default_color = (255,255,255)
    color_list = [x["color"] for x in VIPSEG_CATEGORIES]
    # color_list = [x["color"] for x in COCO_CATEGORIES]
    category_list = [x["name"] for x in VIPSEG_CATEGORIES]
    num_color = len(color_list)
    frame, s_c, s_h, s_w = samples.tensor.shape
    samples = samples.tensor.reshape(1, s_c, -1, s_w)
    for i in range(len(gt_targets)):
        # import pdb
        # pdb.set_trace()
        # image = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() * std + mean # (H, W, 3)
        image = samples[i].permute((1, 2, 0)).cpu().numpy() * std + mean # (H, W, 3)
        # image = unorm(samples[i]).permute((1, 2, 0)).cpu().numpy() * 255 # (H, W, 3)
        # import pdb
        # pdb.set_trace()
        visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
        # input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        original_image = copy.deepcopy(image)
        target = gt_targets[i]
        masks = target["masks"].cpu().numpy()
        num_inst = masks.shape[0]
        zero_base = np.zeros((image.shape))
        for j in range(num_inst):
            # if int(target["labels"][j]) == 60:
            mask = target["masks"][j].cpu().float().numpy() # (H, W)
            if mask.shape != image.shape[:-1]:
                ori_h, ori_w = mask.shape
                mask_new = np.zeros((image.shape[:-1]))
                mask_new[:ori_h, :ori_w] = mask
            else:
                mask_new = mask
            # image[:, :, -1] += 128 * mask_new
            # if "inst_id" in target and target["inst_id"][j] != -1:
            #     color = color_list[target["inst_id"][j] % num_color]
            # else:
            #     color = default_color
            # want to expand mask_new with one more axis in the last and repeat 3 times
            # mask_new = np.expand_dims(mask_new, axis=-1)
            # mask_new = np.repeat(mask_new, 3, axis=-1)
            color = color_list[target["labels"][j]]
            category = category_list[target["labels"][j]]
            # mask_id = target["seg_id"][j]
            # import pdb
            # pdb.set_trace()
            color = [x / 255 for x in color]
            # text = "%s_%d"%(category, mask_id)
            text = "%s"%(category)
            # color = np.array(color)
            # # expand "color" with two more axis front and repeat first axis H times and second axis W times
            # color = np.expand_dims(color, axis=0)
            # color = np.expand_dims(color, axis=0)
            # H,W = mask_new.shape[:2]
            # color = np.repeat(color, H, axis=0)
            # color = np.repeat(color, W, axis=1)
            # zero_base = np.where(mask_new, color, zero_base)
            visualizer.draw_binary_mask(
                mask, color=color, text=text)
        # cv2.imwrite("input_visualize/video_%d/frame_%d_mask.jpg"%(dist.get_rank(), i), zero_base)
        # cv2.imwrite("input_visualize/video_%d/frame_%d_input.jpg"%(dist.get_rank(), i), original_image)
        target["video_name"] = 'test'
        os.makedirs("input_visualize_for_cocnat/video_%d"%dist.get_rank(), exist_ok=True)
        # visualizer.output.save("input_visualize_for_cocnat/video_%d/video_%s_file_%s_mask_%d.jpg"%(
        #     dist.get_rank(), target["video_name"], target["file_name"].split('/')[-1].split('.')[0], i))
        visualizer.output.save("input_visualize_for_cocnat/video_%d/video_%s__mask_%d.jpg"%(
            dist.get_rank(), target["video_name"], i))
        # cv2.imwrite("input_visualize_for_cocovid_new/video_%d/frame_%d.jpg"%(dist.get_rank(), i), original_image)
        # cv2.imwrite("rank_%02d_batch_%d_mask.jpg"%(dist.get_rank(), i), input_mask)
    time.sleep(5)
    sys.exit(0)



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor