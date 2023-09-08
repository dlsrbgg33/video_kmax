import itertools
import os

from typing import List, Optional

import torch
import numpy as np
import tempfile
from collections import OrderedDict
from PIL import Image
from tabulate import tabulate
import json
import contextlib

import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOPanopticEvaluator

from detectron2.utils.visualizer import ColorMode, Visualizer
from video_kmax_deeplab.utils.video_panoptic_visualizer import ColorMode, Visualizer

import io
import math
from PIL import Image

from detectron2.solver.lr_scheduler import _get_warmup_factor_at_iter

import logging
from tqdm import tqdm
import copy

# load video evaluators
import video_kmax_deeplab.evaluation.video_evaluators.eval_vpq as vpq
import video_kmax_deeplab.evaluation.video_evaluators.eval_stq as stq 

logger = logging.getLogger(__name__)


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


class TF2WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used in TF DeepLab2.
    Reference: https://github.com/google-research/deeplab2/blob/main/trainer/trainer_utils.py#L23
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        if self.last_epoch < self.warmup_iters:
            return [
            base_lr * warmup_factor
            for base_lr in self.base_lrs
        ]
        else:
            return [
                base_lr * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                for base_lr in self.base_lrs
            ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class VideoPanopticEvaluatorwithVis(COCOPanopticEvaluator):
    """
    Video Panoptic Evaluator that supports saving visualizations.
        - metrics: STQ or/and VPQ [VPQ takes longer] 
        - visualization: raw panoptic segmentation for server, panoptic segmentation with RGB colored
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None, save_vis_num=0, save_raw_panoptic=False):
        super().__init__(dataset_name=dataset_name, output_dir=output_dir)
        self.metadata = MetadataCatalog.get("vipseg_val_video_panoptic")
        self.output_dir = output_dir
        self.save_vis_num = save_vis_num
        self.save_raw_panoptic = save_raw_panoptic
        
        # currently, hard-coded to choose metrics
        self.metrics = ["STQ", "VPQ"]


    def gen_raw_panoptic(self, panoptic_img, segments_info, video_name, file_name):
        unique_ids = np.unique(panoptic_img)
        H,W = panoptic_img.shape
        ref_image = np.zeros((H,W,3)) 
        new_image = np.zeros((H,W,3))
        segment_ids = [x["id"] for x in segments_info]
        segment_category = [x["category_id"] for x in segments_info]
        segment_isthing = [x["isthing"] for x in segments_info]

        for id in unique_ids:
            if id != 0:
                binary_mask = panoptic_img == id
                binary_mask = np.expand_dims(binary_mask, axis=2)
                binary_mask = np.repeat(binary_mask, repeats=3, axis=2)
                category = segment_category[segment_ids.index(int(id))]
                inst_id = int(id) % self._metadata.label_divisor
                is_thing = segment_isthing[segment_ids.index(int(id))]
                if not is_thing:
                    new_image[:,:,0] = (category + 1)
                    new_image[:,:,1] = 0
                    new_image[:,:,2] = 0
                else:
                    new_image[:,:,0] = (category + 1)
                    new_image[:,:,1] = inst_id // 256
                    new_image[:,:,2] = inst_id % 256    
                ref_image = np.where(binary_mask, new_image, ref_image)
        return ref_image    

        
    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        cur_save_num = 0
        color_dict = {}

        each_video_dict = {}
        each_video_dict["annotations"] = []

        input = inputs[0]
        for idx, output in enumerate(outputs):            
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_seg = panoptic_img.cpu()
            panoptic_img = panoptic_seg.numpy()

            video_name = input["video_id"][idx]
            file_name = os.path.basename(input["file_names"][idx])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            # if cur_save_num < self.save_vis_num:
            image = output["original_image"]
            image = image.permute(1, 2 ,0).cpu().numpy()#[:, :, ::-1]
            # image = self.unorm(image).permute((1, 2, 0)).cpu().numpy() * 255 # (H, W, 3)
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == 0:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label),
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                # panoptic_img += 1
                
            if self.save_raw_panoptic:
                raw_panoptic = self.gen_raw_panoptic(panoptic_img, segments_info, video_name, file_name)
                if not os.path.exists(os.path.join(self.output_dir, 'vis_raw', video_name)):
                    os.makedirs(os.path.join(self.output_dir, 'vis_raw', video_name))
                out_filename = os.path.join(self.output_dir, 'vis_raw', video_name, file_name_png)
                raw_panoptic_image = Image.fromarray(raw_panoptic.astype(np.uint8))
                raw_panoptic_image.save(out_filename)

            visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg, segments_info, color_dict
            )
            if not os.path.exists(os.path.join(self.output_dir, 'vis', video_name)):
                os.makedirs(os.path.join(self.output_dir, 'vis', video_name))
            out_filename = os.path.join(self.output_dir, 'vis', video_name, file_name_png)
            vis_output.save(out_filename)
            vis_output_pil = Image.frombytes('RGB', vis_output.canvas.get_width_height(), vis_output.canvas.tostring_rgb())
            
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                each_video_dict["annotations"].append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )
        each_video_dict["video_id"] = video_name
        each_video_dict["panop_path"] = os.path.join(self.output_dir, 'vis')
        self._predictions.append(each_video_dict)


    def merge_results(self, results):
        # have one OrderedDict named as "video_panoptic_seg"
        # all results have dictionary value according to that key
        merged_results = copy.deepcopy(results[0])
        for key in merged_results.keys():
            merged_results[key] = {}
            for result in results:
                merged_results[key].update(result[key])
        return merged_results

        
    def evaluate(self):
        comm.synchronize()
        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        # len(self._predictions) = num of videos
        # keys of self._predictions[0]: "annotations", "video_id"
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="video_panoptic_eval") as pred_dir:
            logger.info("Writing all video panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                for ann in p["annotations"]:
                    os.makedirs(os.path.join(pred_dir, p["video_id"]), exist_ok=True)
                    with open(os.path.join(pred_dir, p["video_id"], ann["file_name"]), "wb") as f:
                        try:
                            f.write(ann.pop("png_string"))
                        except:
                            print(p["video_id"])
                            print(p["file_name"])
                            import pdb
                            pdb.set_trace()

            # Here, we implement two metrics (VPQ and STQ)
            # VPQ: Video Panoptic Quality
            # STQ: Semantic Tracking Quality
            metric_pass = 0

            if "STQ" in self.metrics:
                results = stq.STQeval(self._predictions, gt_json, pred_dir, gt_folder, self.output_dir)
                metric_pass += 1
            if "VPQ" in self.metrics:
                results = vpq.VPQeval(self._predictions, gt_json, pred_dir, gt_folder, self.output_dir)
                metric_pass += 1

            # if metric_pass == 0:
            #     raise ValueError("No metric is selected for evaluation")
            # if metric_pass > 1:
            #     # where we need merge the results
            #     results = self.merge_results(results)

        return results

class COCOPanopticEvaluatorwithVis(COCOPanopticEvaluator):
    """
    COCO Panoptic Evaluator that supports saving visualizations.
    TODO(qihangyu): Note that original implementation will also write all predictions to a tmp folder
        and then run official evaluation script, we may also check how to copy from the tmp folder for visualization.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None, save_vis_num=0):
        super().__init__(dataset_name=dataset_name, output_dir=output_dir)
        self.metadata = MetadataCatalog.get("coco_2017_val_panoptic_with_sem_seg")
        self.output_dir = output_dir
        self.save_vis_num = save_vis_num


    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        cur_save_num = 0
        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_seg = panoptic_img.cpu()
            panoptic_img = panoptic_seg.numpy()

            file_name = os.path.basename(input["file_names"][0])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            if cur_save_num < self.save_vis_num:
                image = output["original_image"]
                image = image.permute(1, 2 ,0).cpu().numpy()#[:, :, ::-1]
                visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg, segments_info
                )
                if not os.path.exists(os.path.join(self.output_dir, 'vis')):
                    os.makedirs(os.path.join(self.output_dir, 'vis'))
                out_filename = os.path.join(self.output_dir, 'vis', file_name_png)
                vis_output.save(out_filename)
                cur_save_num += 1

            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        # "image_id": input["image_id"][0],
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from video_kmax_deeplab.evaluation.panoptic_evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


def _print_video_panoptic_results(vpq_res):
    headers = ["", "score"]
    data = []
    for name in ["VPQ all", "VPQ thing", "VPQ stuff"]:
        row = [name] + [vpq_res[name]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Video Panoptic Evaluation Results:\n" + table)
