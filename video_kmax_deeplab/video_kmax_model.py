# Reference: Video-kMaX (https://arxiv.org/pdf/2304.04694.pdf)


import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict, Optional
import copy
from tqdm import tqdm

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import EventStorage

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


# inherit from kmax_model 
from .kmax_model import kMaXDeepLab 

# load associater
from .utils.associater.video_stitching_module import VideoStitching
from .utils.associater.lamb_module import LAMB

# debugging module to check input training images
from .utils.debugging import *
from torch.cuda.amp import autocast

SMALL_CONSTANTS = -9999



@META_ARCH_REGISTRY.register()
class VideokMaX(kMaXDeepLab):
    """
        Video-kMaX basically inherits functions from kMaX-DeepLab
        Reference: kMaX-DeepLab (https://arxiv.org/pdf/2207.04044.pdf)
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        class_threshold_thing: float,
        class_threshold_stuff: float,
        overlap_threshold: float,
        reorder_class_weight: float,
        reorder_mask_weight: float,
        thing_area_limit: int,
        stuff_area_limit: int,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        # (currently only supports semantic and panoptic)
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        input_shape: List[int],
        use_clip_stitching: bool,
        memory_name: str,
        num_frames: int,
        test_num_frames: int,
        cfg,
        debug_only,
        backbone_freeze,
        post_processing_seg
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            class_threshold_thing=class_threshold_thing,
            class_threshold_stuff=class_threshold_stuff,
            overlap_threshold=overlap_threshold,
            reorder_class_weight=reorder_class_weight,
            reorder_mask_weight=reorder_mask_weight,
            thing_area_limit=thing_area_limit,
            stuff_area_limit=stuff_area_limit,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # inference
            semantic_on=semantic_on,
            panoptic_on=panoptic_on,
            instance_on=instance_on,
            test_topk_per_image=test_topk_per_image,
            input_shape=input_shape,
        )
        """
        Added Args from kMaX-DeepLab:
            use_clip_stitching: bool, whether to use IoU stitching between clips
            num_frames: int, the number of frames for training.
            test_num_frames: int, the number of frames used for each clip.
            memory_name: str, the name of memory used for long-term tracking
        """

        # config for No. frames
        self.num_frames = num_frames 
        self.test_num_frames = test_num_frames
        
        self.label_divisor = metadata.label_divisor
        
        # Association module
        self.use_clip_stitching = use_clip_stitching
        self.memory_name = memory_name 
        if self.use_clip_stitching:
            self.clip_stitcher = VideoStitching(cfg, test_num_frames, metadata)
        self.lamb = LAMB(cfg, metadata)
        
        # config for debugging input images
        self.debug_only = debug_only

        # whether to freeze the backbone or not 
        # (when training for larger backbones, e.g., ConvNeXt-V2)
        self.backbone_freeze = backbone_freeze

        # select which post-processing to use
        self.post_processing_seg = post_processing_seg


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # config for No. frames
        num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        test_num_frames = cfg.INPUT.TEST_SAMPLING_FRAME_NUM

        # config for association
        use_clip_stitching = cfg.MODEL.VIDEO_KMAX.TEST.USE_CLIP_STITCHING
        memory_name = cfg.MODEL.VIDEO_KMAX.TEST.MEMORY_NAME

        # Loss parameters:
        deep_supervision = cfg.MODEL.KMAX_DEEPLAB.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.KMAX_DEEPLAB.NO_OBJECT_WEIGHT
        share_final_matching = cfg.MODEL.KMAX_DEEPLAB.SHARE_FINAL_MATCHING

        # loss weights
        class_weight = cfg.MODEL.KMAX_DEEPLAB.CLASS_WEIGHT
        dice_weight = cfg.MODEL.KMAX_DEEPLAB.DICE_WEIGHT
        mask_weight = cfg.MODEL.KMAX_DEEPLAB.MASK_WEIGHT
        insdis_weight = cfg.MODEL.KMAX_DEEPLAB.INSDIS_WEIGHT
        aux_semantic_weight = cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_WEIGHT

        debug_only = cfg.INPUT.DEBUG_ONLY
        backbone_freeze = cfg.SOLVER.BACKBONE_FREEZE
        post_processing_seg = cfg.MODEL.KMAX_DEEPLAB.POST_PROCESSING_SEG 

        # building criterion
        matcher = HungarianMatcher()

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
        "loss_pixel_insdis": insdis_weight, "loss_aux_semantic": aux_semantic_weight}

        if deep_supervision:
            dec_layers = sum(cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.DEC_LAYERS)
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        if insdis_weight > 0:
            losses += ["pixels"]
        if aux_semantic_weight > 0:
            losses += ["aux_semantic"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            share_final_matching=share_final_matching,
            pixel_insdis_temperature=cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_TEMPERATURE,
            pixel_insdis_sample_k=cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_SAMPLE_K,
            aux_semantic_temperature=cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_TEMPERATURE,
            aux_semantic_sample_k=cfg.MODEL.KMAX_DEEPLAB.UX_SEMANTIC_SAMPLE_K
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.KMAX_DEEPLAB.TEST.OBJECT_MASK_THRESHOLD,
            "class_threshold_thing": cfg.MODEL.KMAX_DEEPLAB.TEST.CLASS_THRESHOLD_THING,
            "class_threshold_stuff": cfg.MODEL.KMAX_DEEPLAB.TEST.CLASS_THRESHOLD_STUFF,
            "overlap_threshold": cfg.MODEL.KMAX_DEEPLAB.TEST.OVERLAP_THRESHOLD,
            "reorder_class_weight": cfg.MODEL.KMAX_DEEPLAB.TEST.REORDER_CLASS_WEIGHT,
            "reorder_mask_weight": cfg.MODEL.KMAX_DEEPLAB.TEST.REORDER_MASK_WEIGHT,
            "thing_area_limit": cfg.MODEL.KMAX_DEEPLAB.TEST.THING_AREA_LIMIT,
            "stuff_area_limit": cfg.MODEL.KMAX_DEEPLAB.TEST.STUFF_AREA_LIMIT,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.KMAX_DEEPLAB.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.KMAX_DEEPLAB.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.KMAX_DEEPLAB.TEST.PANOPTIC_ON
                or cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.KMAX_DEEPLAB.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.KMAX_DEEPLAB.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "input_shape": cfg.INPUT.IMAGE_SIZE,
            "use_clip_stitching": use_clip_stitching,
            "memory_name": memory_name,          
            "num_frames": num_frames,
            "test_num_frames": test_num_frames,
            "cfg": cfg,
            "debug_only": debug_only,
            "backbone_freeze": backbone_freeze,
            "post_processing_seg": post_processing_seg 
        }


    @property
    def device(self):
        return self.pixel_mean.device


    def preprocess_video(self, batched_inputs, is_train=True):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                normalized_frame = (frame.to(self.device) - self.pixel_mean) / self.pixel_std
                images.append(normalized_frame)
        return images


    def height_wise_concat(self, input):
        return torch.cat(torch.split(input, 1), dim=2)


    def target_concat_height_wise(self, targets):
        total_dataset_dict = []
        new_dataset_dict = {}
        new_dataset_dict["semantic_masks"] = []
        new_dataset_dict["labels"] = [] 
        new_dataset_dict["masks"] = [] 
        seg_to_idx = {}
        for idx, target in enumerate(targets):
            new_dataset_dict["semantic_masks"].append(target["semantic_masks"]) 
            if idx == 0:
                for mask_idx, seg_id in enumerate(target["seg_id"]):
                    current_mask = target["masks"][mask_idx]
                    setting = torch.zeros((current_mask.repeat(self.num_frames, 1).shape)).to(current_mask).bool()
                    setting[:current_mask.shape[0], :] = current_mask
                    seg_to_idx[int(seg_id)] = {
                        "labels": target["labels"][mask_idx],
                        "masks": setting,
                    }
            else:
                for mask_idx, seg_id in enumerate(target["seg_id"]):
                    current_mask = target["masks"][mask_idx]
                    current_mask_shape = current_mask.shape[0]
                    if int(seg_id) in seg_to_idx:
                        seg_to_idx[int(seg_id)]["masks"][idx * current_mask_shape:(idx + 1) * current_mask_shape, :] = target["masks"][mask_idx]
                    else:
                        setting = torch.zeros((current_mask.repeat(self.num_frames, 1).shape)).to(current_mask).bool()
                        setting[idx * current_mask_shape:(idx + 1) * current_mask_shape, :] = current_mask
                        seg_to_idx[int(seg_id)] = {
                            "labels": target["labels"][mask_idx],
                            "masks": setting,
                        }

        for seg_id in seg_to_idx.keys():
            new_dataset_dict["masks"].append(seg_to_idx[seg_id]["masks"])
            new_dataset_dict["labels"].append(int(seg_to_idx[seg_id]["labels"]))

        new_dataset_dict["semantic_masks"] = torch.cat(new_dataset_dict["semantic_masks"], dim=0)
        new_dataset_dict["masks"] = torch.stack(new_dataset_dict["masks"], dim=0)

        new_dataset_dict["labels"] = torch.Tensor(new_dataset_dict["labels"]).int().to(new_dataset_dict["semantic_masks"])
        total_dataset_dict.append(new_dataset_dict)

        return total_dataset_dict


    def _output_height_concat(self, outputs):
        for out_key in list(outputs.keys()):
            if out_key in ["pred_masks", "pixel_feature"]:
                outputs[out_key] = torch.cat(torch.split(outputs[out_key], 1, 0), dim=2)
        
            if out_key == "aux_outputs":
                for aux_output in outputs[out_key]:
                    for aux_key in list(aux_output.keys()):
                        aux_output[aux_key] = torch.cat(torch.split(aux_output[aux_key], 1, 0), dim=2)
            if out_key == "aux_semantic_pred":
                outputs[out_key] = torch.cat(torch.split(outputs[out_key], 1, 0), dim=2)
                

    def _integrate_into_panoptic_seg(self, panoptic_r):
        panoptic_seg, segments_info = panoptic_r[0], panoptic_r[1]
        segments_info_copy = copy.deepcopy(segments_info)
        for segment_info in segments_info_copy:
            binary_mask = panoptic_seg == segment_info["id"]
            panoptic_seg = torch.where(
                binary_mask, panoptic_seg + segment_info["category_id"] * self.label_divisor, panoptic_seg)
        return panoptic_seg


    def _mask_to_bbox(self, binary_mask):
        # binary_mask is a tensor of shape [THxW]
        # first, convert it to [TxHxW] and add it to the first channel to obtain [HxW]
        _, W = binary_mask.shape
        binary_mask = binary_mask.reshape(self.test_num_frames, -1, W)
        binary_mask = binary_mask.sum(dim=0)
        H,W = binary_mask.shape
        # then, find the bbox that is surrounding the binary_mask where the value is greater_or_equal than 1
        # the bbox is in the format of [xtl, ytl, xbr, ybr]
        y, x = torch.where(binary_mask >= 1)
        bbox = torch.stack([x.min()/W, y.min()/H , x.max()/W, y.max()/H])
        return bbox
    

    def _get_asso_dict(self, panoptic_seg, segment_info):
        segment_info_copy = segment_info
        aux_dict = {}
        for segment in segment_info_copy:
            if segment["isthing"] == 1:
                id = segment["id"] + segment["category_id"] * self.label_divisor
                binary_mask = panoptic_seg == id
                if binary_mask.sum() == 0:
                    continue
                bbox_xyxy = self._mask_to_bbox(binary_mask)
                aux_dict[id] = {"class_conf": segment["class_conf"],
                                "cluster_feat": segment["cluster_feat"],
                                "bbox_xyxy": bbox_xyxy}
        return aux_dict


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
                concated into clips
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (num_frames*height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        

        if self.training:
            images = self.preprocess_video(batched_inputs)
            if "is_real_pixels" in batched_inputs[0]:
                is_real_pixels = []
                for video in batched_inputs:
                    is_real_pixels += [x for x in video["is_real_pixels"]]
                # Set all padded pixel values to 0.
                images = [x * y.to(x) for x, y in zip(images, is_real_pixels)]

            # We perform zero padding to ensure input shape equal to self.input_shape.
            # The padding is done on the right and bottom sides. 
            # images_padded = copy.deepcopy(images)
            images_padded = images
            for idx in range(len(images)):
                cur_height, cur_width = images[idx].shape[-2:]
                padding = (0, max(0, self.input_shape[1] - cur_width), 0, max(0, self.input_shape[0] - cur_height), 0, 0)
                images_padded[idx] = F.pad(images[idx], padding, value=0)
            images = ImageList.from_tensors(images, -1)
            images_padded = ImageList.from_tensors(images_padded, -1)

            # mask classification target
            if "instances" in batched_inputs[0]:
                for video in batched_inputs:
                    gt_instances = [x.to(self.device) for x in video["instances"]]
                    gt_semantic = [x.to(self.device) for x in video["sem_seg_gt"]]
                    gt_video_file = [x for x in video["video_id"]]
                    gt_image_file = [x for x in video["file_names"]]
                targets = self.prepare_targets(gt_instances, gt_semantic, images_padded, gt_video_file, gt_image_file)
            else:
                targets = None

            targets = self.target_concat_height_wise(targets)

            # we can check whether we have expecting training images and ground truth.
            if self.debug_only:
                debug(images_padded, targets)

            # training and concat the pedictions in height-wise concat manner
            input_tensor = images_padded.tensor
            input_tensor = input_tensor.to(memory_format=torch.channels_last)
            features = self.backbone(input_tensor)
            outputs = self.sem_seg_head(features)
            self._output_height_concat(outputs) 

            with autocast(enabled=False):
                # bipartite matching-based loss
                for output_key in ["pixel_feature", "pred_masks", "pred_logits", "aux_semantic_pred"]:
                    if output_key in outputs:
                        outputs[output_key] = outputs[output_key].float()
                for i in range(len(outputs["aux_outputs"])):
                    for output_key in ["pixel_feature", "pred_masks", "pred_logits"]:
                        outputs["aux_outputs"][i][output_key] = outputs["aux_outputs"][i][output_key].float()

                with EventStorage():  # capture events in a new storage to discard them
                    losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                
                return losses
        else: 
            return self.forward_inference(batched_inputs)

    def forward_inference(self, batched_inputs):
        # define the memory for each video
        memory_dict = {}

        processed_results = []
        # sequetially loading clip from video
        if self.test_num_frames > 1:
            load_clip_gap = self.test_num_frames - 1
        else:
            load_clip_gap = 1
        
        expand_batched_inputs = copy.deepcopy(batched_inputs)
        # expand the batched_inputs to the length of self.test_num_frames
        self._input_expansion(expand_batched_inputs)

        stuff_video_memory = {}        
        for i in range(0, len(batched_inputs[0]["image"]), load_clip_gap):
            # generate the clip as new_batched_input
            new_batched_input = [{}]
            new_batched_input[0]["image"] = expand_batched_inputs[0]["image"][i:i+self.test_num_frames]
            new_batched_input[0]["height"] = expand_batched_inputs[0]["height"][i:i+self.test_num_frames]
            new_batched_input[0]["width"] = expand_batched_inputs[0]["width"][i:i+self.test_num_frames]

            images = self.preprocess_video(new_batched_input)

            # We perform zero padding to ensure input shape equal to self.input_shape.
            # The padding is done on the right and bottom sides. 
            
            images_padded = copy.deepcopy(images)
            for idx in range(len(images)):
                cur_height, cur_width = images[idx].shape[-2:]
                padding = (0, max(0, self.input_shape[1] - cur_width), 0, max(0, self.input_shape[0] - cur_height), 0, 0)
                images_padded[idx] = F.pad(images[idx], padding, value=0)
            images = ImageList.from_tensors(images, -1)
            images_padded = ImageList.from_tensors(images_padded, -1)
            
            # clip tensor for training or inference
            test_clip_tensor = images_padded.tensor
            # height-axis concat for evaluation
            test_clip_height = torch.stack(new_batched_input[0]["image"], dim=0).to(self.device)

            features = self.backbone(test_clip_tensor)
            outputs = self.sem_seg_head(features)
    
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            cluster_results = outputs["cluster_feature"]
            
            align_corners = (test_clip_tensor.shape[-1] % 2 == 1) 
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(test_clip_tensor.shape[-2], test_clip_tensor.shape[-1]),
                mode="bilinear",
                align_corners=align_corners)
            
            m_b, m_k, m_h, m_w = mask_pred_results.shape
            # mask_pred_results has the shape [BTxKxHxW], but want to split into [BxTxKxHxW]
            mask_pred_results = mask_pred_results.view(
                -1, self.test_num_frames, m_k, m_h, m_w
            )

            del outputs
            
            for mask_cls_result, mask_pred_result, input_per_image, image_size, cluster_result in zip(
                mask_cls_results, mask_pred_results, new_batched_input, images.image_sizes, cluster_results
            ): 
                height = input_per_image.get("height", image_size[0])[0]
                width = input_per_image.get("width", image_size[1])[0]

                # cur_image and mask_pred_result have the shape [TxHxW]
                cur_image = test_clip_height[:, :, :image_size[0], :image_size[1]]
                mask_pred_result = mask_pred_result[:, :, :image_size[0], :image_size[1]]
                mask_pred_result = F.interpolate(
                    mask_pred_result, size=(height, width), mode="bilinear", align_corners=align_corners
                )
                height_wised_results = self.height_wise_concat(mask_pred_result)[0]

                cur_image = F.interpolate(
                    cur_image.float(), size=(height, width), mode="bilinear", align_corners=align_corners
                )
                cur_image = self.height_wise_concat(cur_image)[0]

                if self.sem_seg_postprocess_before_inference:
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    if self.post_processing_seg == "pixel_wise":
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference_pixel)(
                            mask_cls_result, height_wised_results, cluster_result, height)
                    elif self.post_processing_seg == "mask_wise":
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference_mask)(
                            mask_cls_result, height_wised_results, cluster_result, height)
                    else:
                        raise NotImplementedError(
                            "Currently, pixel and mask wise post-processing are only supported.")
                    # panoptic_r includes "panoptic_seg", "segments_info".
                    # "segments_info" is the dict with ['id', 'category_id', 'isthing', 'score']
                    # combine the panoptic_seg and segments_info into one [HxW],
                    # each pixel is label_divisor * category_id + id
                    panoptic_seg = self._integrate_into_panoptic_seg(panoptic_r)
                    
                    # obtain the association dictionary ("class_conf", "cluster_feat", "bbox_xyxy")
                    asso_dict_feat = self._get_asso_dict(panoptic_seg, panoptic_r[1])

                    # video stitching 
                    if self.use_clip_stitching:
                        (stitched_panoptic_seg, non_match_dict,
                         matched_cur_to_prev, asso_dict_feat) = self.clip_stitcher(
                            panoptic_seg, asso_dict_feat, i, height)
                        if self.memory_name is not None:
                            # need to remove stuff ids from dictionary for thing association
                            self.lamb.remove_stuff([non_match_dict, matched_cur_to_prev])
                            # non_stitched_ids: consists of id and cluster_feat that are not matched
                            non_stitched_ids_feat = self.clip_stitcher.update_aux_ids(
                                asso_dict_feat, non_match_dict, i)
                            stitched_ids_feat = self.clip_stitcher.update_aux_ids(
                                asso_dict_feat, matched_cur_to_prev, i)
                                
                            # need to implement memory system for unmathced ids
                            # restitching the stitched_ids based on memory_dict
                            stitched_panoptic_seg = self.lamb.memory_decode_and_encode(
                                stitched_panoptic_seg, non_stitched_ids_feat, stitched_ids_feat,
                                asso_dict_feat, matched_cur_to_prev, memory_dict, i)
                            self.clip_stitcher.save_stitched_panoptic_to_buffer(stitched_panoptic_seg)
                        
                        panoptic_seg_after_process = stitched_panoptic_seg
                        # print(i)
                    else:
                        panoptic_seg_after_process = panoptic_seg

                    panoptic_seg_after_process = self.stuff_id_merge(panoptic_seg_after_process, stuff_video_memory)

                    # split the stitched_panoptic_seg into test_num_frames to save them separately for evaluation
                    split_shape = panoptic_seg_after_process.shape[0]//(self.test_num_frames)
                    panoptic_seg_after_process = torch.split(panoptic_seg_after_process, split_shape, dim=0)
                    cur_image = torch.split(cur_image, split_shape, dim=1)

                    # save the processed results
                    # clip #1 (frame 1, frame 2) and clip #2 (frame3, frame4)
                    if i % self.test_num_frames == 0:
                        for panoptic_split, cur_image_split in zip(panoptic_seg_after_process, cur_image):
                            processed_results.append({})
                            processed_results[-1]["panoptic_seg"] = [panoptic_split, None]
                            processed_results[-1]["original_image"] = cur_image_split
                    # processed_results.append({})
                    # processed_results[-1]["panoptic_seg"] = [panoptic_seg_after_process[0], None]
                    # processed_results[-1]["original_image"] = cur_image[0]


        processed_results = processed_results[:len(batched_inputs[0]["image"])]
        return processed_results

    def _input_expansion(self, batched_inputs):
        batched_inputs[0]["image"] += [batched_inputs[0]["image"][-1]] * (self.test_num_frames-1)
        batched_inputs[0]["height"] += [batched_inputs[0]["height"][-1]] * (self.test_num_frames-1)
        batched_inputs[0]["width"] += [batched_inputs[0]["width"][-1]] * (self.test_num_frames-1)

    def stuff_id_merge(self, panoptic_seg, stuff_video_memory):
        # panoptic_seg: [HxW]
        # stuff_video_memory: {stuff_category: stuff_panoptic_id}

        panoptic_seg_copy = copy.deepcopy(panoptic_seg)
        panoptic_seg_copy[panoptic_seg_copy==0] = -self.label_divisor
        # panoptic_seg_copy = torch.where(panoptic_seg_copy==0, -1, panoptic_seg_copy)

        class_label = panoptic_seg_copy // self.label_divisor
        class_label_unique = class_label.unique()
        
        if len(stuff_video_memory) > 0:
            for id in stuff_video_memory.keys():
                panoptic_seg_class = panoptic_seg_copy // self.label_divisor
                if id in panoptic_seg_class:
                    panoptic_seg_copy[panoptic_seg_class == id] = stuff_video_memory[id]
        
        for label in class_label_unique:
            if int(label) >= 0:
                if label not in self.metadata.thing_dataset_id_to_contiguous_id.values():
                    mask = class_label == int(label)
                    panoptic_id = panoptic_seg_copy[mask][0]
                    if int(label) not in stuff_video_memory.keys():
                        stuff_video_memory[int(label)] = panoptic_id

        panoptic_seg_copy[panoptic_seg_copy==-self.label_divisor] = 0

        return panoptic_seg_copy


    def prepare_targets(self, targets, targets_semantic, images, video_names, file_names):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, semantic_gt_mask, video_name, file_name in zip(targets, targets_semantic, video_names, file_names):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            new_targets.append(
                {   "labels": targets_per_image.gt_classes,
                    "masks": gt_masks,
                    "seg_id": targets_per_image.seg_id,
                    "semantic_masks": semantic_gt_mask,
                    "video_name": video_name,
                    "file_name": file_name,
                }
            )
        return new_targets


    def semantic_inference(self, mask_cls, mask_pred):
        # For cls prob, we exluced the void class following
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = F.softmax(mask_pred, dim=0)
        # Mask2Former combines the soft prob to obtain sem results
        # In kMaX, the mask logits is argmax'ed.
        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L315
        mask_pred = (mask_pred >= mask_pred.max(dim=0, keepdim=True)[0]).float()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg


    def panoptic_inference_mask(self, mask_cls, mask_pred, cluster_results, height):
        # mask_cls: N x C
        # mask_pred: N x TH x W
        # cluster_results: D x N
        # some hyper-params
        # mask_cls = mask_cls[:, list(self.metadata.sorted_redirect_dict.values())]
        num_mask_slots = mask_pred.shape[0]
        cls_threshold_thing = self.class_threshold_thing
        cls_threshold_stuff = self.class_threshold_stuff
        object_mask_threshold = self.object_mask_threshold
        overlap_threshold = self.overlap_threshold
        reorder_class_weight = self.reorder_class_weight
        reorder_mask_weight = self.reorder_mask_weight
        
        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L675
        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L199
        cls_scores, cls_labels = F.softmax(mask_cls, dim=-1)[..., :-1].max(-1) # N
        mask_scores = F.softmax(mask_pred, dim=0)
        binary_masks = mask_scores > object_mask_threshold # N x H x W
        mask_scores_flat = mask_scores.flatten(1) # N x HW # 각 픽셀들의 마스크 점수
        binary_masks_flat = binary_masks.flatten(1).float() # N x HW # 각 픽셀들이 마스크에 속하는지 여부
        pixel_number_flat = binary_masks_flat.sum(1) # N # 그 마스크에 해당하는 픽셀들의 개수
        # mask_scores_flat: 각각의 마스크에 해당하는 픽셀들의 마스크 점수들의 평균
        mask_scores_flat = (mask_scores_flat * binary_masks_flat).sum(1) / torch.clamp(pixel_number_flat, min=1.0) # N

        reorder_score = (cls_scores ** reorder_class_weight) * (mask_scores_flat ** reorder_mask_weight) # N
        reorder_indices = torch.argsort(reorder_score, dim=-1, descending=True)

        panoptic_seg = torch.zeros((mask_pred.shape[1], mask_pred.shape[2]),
         dtype=torch.int32, device=mask_pred.device)
        segments_info = []

        current_segment_id = 0
        stuff_memory_list = {}

        for i in range(num_mask_slots):
            cur_idx = reorder_indices[i].item() # 1
            cur_binary_mask = binary_masks[cur_idx] # H x W
            cur_cls_score = cls_scores[cur_idx].item() # 1 
            cur_cls_label = cls_labels[cur_idx].item() # 1
            cur_cluster = cluster_results[:, cur_idx]
            
            is_thing = cur_cls_label in self.metadata.thing_dataset_id_to_contiguous_id.values()
            is_confident = (is_thing and cur_cls_score > cls_threshold_stuff) or (
                (not is_thing) and cur_cls_score > cls_threshold_stuff)

            original_pixel_number = cur_binary_mask.float().sum()
            new_binary_mask = torch.logical_and(cur_binary_mask, (panoptic_seg == 0))
            new_pixel_number = new_binary_mask.float().sum()
            is_not_overlap_too_much = new_pixel_number > (original_pixel_number * overlap_threshold)

            # Filter by area size.
            if (is_thing and new_pixel_number < self.thing_area_limit) or (
                not is_thing and new_pixel_number < self.stuff_area_limit):
                continue


            new_binary_mask_split = torch.split(new_binary_mask, height, dim=0)
            
            # new_binary_mask_list = []
            # for new_binary in new_binary_mask_split:
            #     new_binary_number = new_binary.float().sum()
            #     if int(new_binary_number) < 100:
            #         new_binary = torch.zeros_like(new_binary)
            #     new_binary_mask_list.append(new_binary)
            # new_binary_mask = torch.cat(new_binary_mask_list, dim=0)
            # if int(new_binary_mask.sum()) == 0:
            #     continue
              
                
            if is_confident and is_not_overlap_too_much:
                # merge stuff regions
                if not is_thing:
                    if int(cur_cls_label) in stuff_memory_list.keys():
                        panoptic_seg[new_binary_mask] = stuff_memory_list[int(cur_cls_label)]
                        continue
                    else:
                        stuff_memory_list[int(cur_cls_label)] = current_segment_id + 1

                current_segment_id += 1                
                panoptic_seg[new_binary_mask] = current_segment_id
                
                # add mask confidence 
                mask_score_max, _ = torch.max(mask_scores, dim=0)

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(is_thing),
                        "category_id": int(cur_cls_label),
                        "class_conf": float(cur_cls_score),
                        "cluster_feat": cur_cluster, # N
                        "mask_confidence": mask_score_max
                    }
                )

        return panoptic_seg, segments_info
    

    def panoptic_inference_pixel(self, mask_cls, mask_pred, cluster_results, height):
        # mask_cls: N x C
        # mask_pred: N x TH x W
        # cluster_results: D x N
        # some hyper-params
        # mask_cls = mask_cls[:, list(self.metadata.sorted_redirect_dict.values())]
        
        num_mask_slots = mask_pred.shape[0]
        cls_threshold_thing = self.class_threshold_thing
        object_mask_threshold = self.object_mask_threshold

        cls_scores, cls_labels = F.softmax(mask_cls, dim=-1)[..., :-1].max(-1) # N
        cls_thresholded = cls_scores > object_mask_threshold
        cls_thresholded = cls_thresholded.unsqueeze(1).unsqueeze(1).to(torch.float64)
        mask_scores = mask_pred * cls_thresholded + (SMALL_CONSTANTS) * (1 - cls_thresholded)
        

        mask_scores_soft = F.softmax(mask_scores, dim=0)
        mask_max_idx = mask_scores.argmax(dim=0)
        mask_max_val = mask_scores_soft.max(dim=0)[0]

        panoptic_seg = torch.zeros((mask_pred.shape[1], mask_pred.shape[2]),
         dtype=torch.int32, device=mask_pred.device)

        segments_info = []
        current_segment_id = 0
        stuff_memory_list = {}
        
        for cur_idx in mask_max_idx.unique():
            cur_binary_mask = mask_max_idx == cur_idx # H x W
            cur_cls_score = cls_scores[cur_idx].item() # 1
            mask_max_val_zero = torch.zeros_like(mask_max_val).to(mask_max_val)
            current_mask_score = torch.where(cur_binary_mask,
                        mask_max_val, mask_max_val_zero)  
            cur_cls_label = cls_labels[cur_idx].item() # 1
            cur_cluster = cluster_results[:, cur_idx]
            
            is_thing = cur_cls_label in self.metadata.thing_dataset_id_to_contiguous_id.values()
            confident_area = current_mask_score > cls_threshold_thing

            if not is_thing:
                if int(cur_cls_label) in stuff_memory_list.keys():
                    panoptic_seg[confident_area] = stuff_memory_list[int(cur_cls_label)]
                    continue
                else:
                    stuff_memory_list[int(cur_cls_label)] = current_segment_id + 1

            current_segment_id += 1                
            panoptic_seg[confident_area] = current_segment_id

            # add mask confidence 
            mask_score_max, _ = torch.max(mask_scores, dim=0)

            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": bool(is_thing),
                    "category_id": int(cur_cls_label),
                    "class_conf": float(cur_cls_score),
                    "cluster_feat": cur_cluster, # N
                    "mask_confidence": mask_score_max
                }
            )

        return panoptic_seg, segments_info
    
