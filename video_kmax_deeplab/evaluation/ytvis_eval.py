# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from kmax_deeplab.data.datasets.ytvis_api.ytvos import YTVOS
from kmax_deeplab.data.datasets.ytvis_api.ytvoseval import YTVOSeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.structures import Instances

# from visualizer import TrackVisualizer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode

class YTVISEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._ytvis_api = YTVOS(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._ytvis_api.dataset

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        prediction = instances_to_coco_json_video(inputs, outputs, self._metadata, self._output_dir)
        self._predictions.extend(prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for YTVIS format ...")

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in predictions:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        coco_eval = (
            _evaluate_predictions_on_coco(
                self._ytvis_api,
                predictions,
            )
            if len(predictions) > 0
            else None  # cocoapi does not handle empty results very well
        )

        res = self._derive_coco_results(
            coco_eval, class_names=self._metadata.get("thing_classes")
        )
        self._results["segm"] = res

    def _derive_coco_results(self, coco_eval, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format("segm") + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format("segm") + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json_video(inputs, outputs, metadata, output_dir):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]
    video_length = len(inputs)

    image_size = outputs["image_size"]
    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]

    ytvis_results = []
    for instance_id, (s, l, m) in enumerate(zip(scores, labels, masks)):
        segms = [
            mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
            for _mask in m
        ]
        for rle in segms:
            rle["counts"] = rle["counts"].decode("utf-8")

        res = {
            "video_id": video_id[0],
            "score": s,
            "category_id": l,
            "segmentations": segms,
        }
        ytvis_results.append(res)

    # import pdb; pdb.set_trace()
    vid_frames = []
    paths = inputs[0]["file_names"][0][:5]
    # paths = inputs[0]["file_names"]
    for path in paths:
        img = read_image(path, format="BGR")
        vid_frames.append(img)
        
    frame_masks = list(zip(*masks))
    total_vis_output = []
    for frame_idx in range(len(vid_frames)):
        frame = vid_frames[frame_idx][:, :, ::-1]
        visualizer = TrackVisualizer(frame, metadata, instance_mode=ColorMode.IMAGE)
        ins = Instances(image_size)
        if len(scores) > 0:
            ins.scores = scores
            ins.pred_classes = labels
            ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)

        vis_output = visualizer.draw_instance_predictions(predictions=ins)
        total_vis_output.append(vis_output)

    output_folder = os.path.join(output_dir, str(video_id))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, _vis_output in enumerate(total_vis_output):
        out_filename = os.path.join(output_folder, f"{idx}.jpg")
        _vis_output.save(out_filename)

    return ytvis_results


def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    img_ids=None,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_results = copy.deepcopy(coco_results)
    # When evaluating mask AP, if the results contain bbox, cocoapi will
    # use the box area as the area of the instance, instead of the mask area.
    # This leads to a different definition of small/medium/large.
    # We remove the bbox field to let mask AP use mask area.
    for c in coco_results:
        c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = YTVOSeval(coco_gt, coco_dt)
    # For COCO, the default max_dets_per_image is [1, 10, 100].
    max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# reference: https://github.com/sukjunhwang/IFC/blob/master/projects/IFC/demo/visualizer.py
import torch
import numpy as np
import matplotlib.colors as mplc

from detectron2.utils.visualizer import ColorMode, GenericMask, Visualizer, _create_text_labels


_ID_JITTERS = [[0.9047944201469568, 0.3241718265806123, 0.33443746665210006], [0.4590171386127151, 0.9095038146383864, 0.3143840671974788], [0.4769356899795538, 0.5044406738441948, 0.5354530846360839], [0.00820945625670777, 0.24099210193126785, 0.15471834055332978], [0.6195684374237388, 0.4020380013509799, 0.26100266066404676], [0.08281237756545068, 0.05900744492710419, 0.06106221202154216], [0.2264886829978755, 0.04925271007292076, 0.10214429345996079], [0.1888247470009874, 0.11275000298612425, 0.46112894830685514], [0.37415767691880975, 0.844284596118331, 0.950471611180866], [0.3817344218157631, 0.3483259270707101, 0.6572989333690541], [0.2403115731054466, 0.03078280287279167, 0.5385975692534737], [0.7035076951650824, 0.12352084932325424, 0.12873080308790197], [0.12607434914489934, 0.111244793010015, 0.09333334699716023], [0.6551607300342269, 0.7003064103554443, 0.4131794512286162], [0.13592107365596595, 0.5390702818232149, 0.004540643174930525], [0.38286244894454347, 0.709142545393449, 0.529074791609835], [0.4279376583651734, 0.5634708596431771, 0.8505569717104301], [0.3460488523902999, 0.464769595519293, 0.6676839675477276], [0.8544063246675081, 0.5041190233407755, 0.9081217697141578], [0.9207009090747208, 0.2403865944739051, 0.05375410999863772], [0.6515786136947107, 0.6299918449948327, 0.45292029442034387], [0.986174217295693, 0.2424849846977214, 0.3981993323108266], [0.22101915872994693, 0.3408589198278038, 0.006381420347677524], [0.3159785813515982, 0.1145748921741011, 0.595754317197274], [0.10263421488052715, 0.5864139253490858, 0.23908000741142432], [0.8272999391532938, 0.6123527260897751, 0.3365197327803193], [0.5269583712937912, 0.25668929554516506, 0.7888411215078127], [0.2433880265410031, 0.7240751234287827, 0.8483215810528648], [0.7254601709704898, 0.8316525547295984, 0.9325253855921963], [0.5574483824856672, 0.2935331727879944, 0.6594839453793155], [0.6209642371433579, 0.054030693198821256, 0.5080873988178534], [0.9055507077365624, 0.12865888619203514, 0.9309191861440005], [0.9914469722960537, 0.3074114506206205, 0.8762107657323488], [0.4812682518247371, 0.15055826298548158, 0.9656340505308308], [0.6459219454316445, 0.9144794010251625, 0.751338812155106], [0.860840174209798, 0.8844626353077639, 0.3604624506769899], [0.8194991672032272, 0.926399617787601, 0.8059222327343247], [0.6540413175393658, 0.04579445254618297, 0.26891917826531275], [0.37778835833987046, 0.36247927666109536, 0.7989799305827889], [0.22738304978177726, 0.9038018263773739, 0.6970838854138303], [0.6362015495896184, 0.527680794236961, 0.5570915425178721], [0.6436401915860954, 0.6316925317144524, 0.9137151236993912], [0.04161828388587163, 0.3832413349082706, 0.6880829921949752], [0.7768167825719299, 0.8933821497682587, 0.7221278391266809], [0.8632760876301346, 0.3278628094906323, 0.8421587587114462], [0.8556499133262127, 0.6497385872901932, 0.5436895688477963], [0.9861940318610894, 0.03562313777386272, 0.9183454677106616], [0.8042586091176366, 0.6167222703170994, 0.24181981557207644], [0.9504247117633057, 0.3454233714011461, 0.6883727005547743], [0.9611909135491202, 0.46384154263898114, 0.32700443315058914], [0.523542176970206, 0.446222414615845, 0.9067402987747814], [0.7536954008682911, 0.6675512338797588, 0.22538238957839196], [0.1554052265688285, 0.05746097492966129, 0.8580358872587424], [0.8540838640971405, 0.9165504335482566, 0.6806982829158964], [0.7065090319405029, 0.8683059983962002, 0.05167128320624026], [0.39134812961899124, 0.8910075505622979, 0.7639815712623922], [0.1578117311479783, 0.20047326898284668, 0.9220177338840568], [0.2017488993096358, 0.6949259970936679, 0.8729196864798128], [0.5591089340651949, 0.15576770423813258, 0.1469857469387812], [0.14510398622626974, 0.24451497734532168, 0.46574271993578786], [0.13286397822351492, 0.4178244533944635, 0.03728728952131943], [0.556463206310225, 0.14027595183361663, 0.2731537988657907], [0.4093837966398032, 0.8015225687789814, 0.8033567296903834], [0.527442563956637, 0.902232617214431, 0.7066626674362227], [0.9058355503297827, 0.34983989180213004, 0.8353262183839384], [0.7108382186953104, 0.08591307895133471, 0.21434688012521974], [0.22757345065207668, 0.7943075496583976, 0.2992305547627421], [0.20454109788173636, 0.8251670332103687, 0.012981987094547232], [0.7672562637297392, 0.005429019973062554, 0.022163616037108702], [0.37487345910117564, 0.5086240194440863, 0.9061216063654387], [0.9878004014101087, 0.006345852772772331, 0.17499753379350858], [0.030061528704491303, 0.1409704315546606, 0.3337131835834506], [0.5022506782611504, 0.5448435505388706, 0.40584238936140726], [0.39560774627423445, 0.8905943695833262, 0.5850815030921116], [0.058615671926786406, 0.5365713844300387, 0.1620457551256279], [0.41843842882069693, 0.1536005983609976, 0.3127878501592438], [0.05947621790155899, 0.5412421167331932, 0.2611322146455659], [0.5196159938235607, 0.7066461551682705, 0.970261497412556], [0.30443031606149007, 0.45158581060034975, 0.4331841153149706], [0.8848298403933996, 0.7241791700943656, 0.8917110054596072], [0.5720260591898779, 0.3072801598203052, 0.8891066705989902], [0.13964015336177327, 0.2531778096760302, 0.5703756837403124], [0.2156307542329836, 0.4139947500641685, 0.87051676884144], [0.10800455881891169, 0.05554646035458266, 0.2947027428551443], [0.35198009410633857, 0.365849666213808, 0.06525787683513773], [0.5223264108118847, 0.9032195574351178, 0.28579084943315025], [0.7607724246546966, 0.3087194381828555, 0.6253235528354899], [0.5060485442077824, 0.19173600467625274, 0.9931175692203702], [0.5131805830323746, 0.07719515392040577, 0.923212006754969], [0.3629762141280106, 0.02429179642710888, 0.6963754952399983], [0.7542592485456767, 0.6478893299494212, 0.3424965345400731], [0.49944574453364454, 0.6775665366832825, 0.33758796076989583], [0.010621818120767679, 0.8221571611173205, 0.5186257457566332], [0.5857910304290109, 0.7178133992025467, 0.9729243483606071], [0.16987399482717613, 0.9942570210657463, 0.18120758122552927], [0.016362572521240848, 0.17582788603087263, 0.7255176922640298], [0.10981764283706419, 0.9078582203470377, 0.7638063718334003], [0.9252097840441119, 0.3330197086990039, 0.27888705301420136], [0.12769972651171546, 0.11121470804891687, 0.12710743734391716], [0.5753520518360334, 0.2763862879599456, 0.6115636613363361]]


class TrackVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(
            img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode
        )
        self.cpu_device = torch.device("cpu")
    
    def _jitter(self, color, id):
        """
        Randomly modifies given color to produce a slightly different color than the color given.
        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.
        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        vec = _ID_JITTERS[id]
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        preds = predictions.to(self.cpu_device)

        boxes = preds.pred_boxes if preds.has("pred_boxes") else None
        scores = preds.scores if preds.has("scores") else None
        classes = preds.pred_classes if preds.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        if labels is not None:
            labels = ["[{}] ".format(_id) + l for _id, l in enumerate(labels)]

        if preds.has("pred_masks"):
            masks = np.asarray(preds.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if classes is None:
            return self.output

        colors = [
            self._jitter([x / 255 for x in self.metadata.thing_colors[c]], id) for id, c in enumerate(classes)
        ]
        alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (preds.pred_masks.any(dim=0) > 0).numpy()
                if preds.has("pred_masks")
                else None
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )

        return self.output
