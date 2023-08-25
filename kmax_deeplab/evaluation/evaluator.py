# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import copy
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

from detectron2.data import MetadataCatalog

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset_ensemble(
    model, model_ref, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    metadata = MetadataCatalog.get("vipseg_val_video_panoptic")
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()

            model_ref.eval()
            outputs = model(inputs, model_ref) 
            outputs_ref = model_ref(inputs, model)

            # video-level aggregation
            # dictionary: {key: value} -> {id: corresponding mask}
            # N:1 / 1:M
            # first for main
            
            id_dict = {}
            id_dict_idx = {}
            for idx_main, output in enumerate(outputs):
                panoptic_seg_ids = torch.unique(output["panoptic_seg"][0])
                for panoptic_seg_id in panoptic_seg_ids:
                    # only consider thing_id
                    panoptic_class = panoptic_seg_id // 10000
                    if panoptic_class in metadata.thing_dataset_id_to_contiguous_id.values():
                        id_mask = output["panoptic_seg"][0] == int(panoptic_seg_id)
                        id_mask = id_mask > 0
                        if int(panoptic_seg_id) in id_dict:
                            id_dict[int(panoptic_seg_id)].append(id_mask)
                            id_dict_idx[int(panoptic_seg_id)].append(idx_main)
                        else:
                            id_dict[int(panoptic_seg_id)] = [id_mask]
                            id_dict_idx[int(panoptic_seg_id)] = [idx_main]
            
            id_dict_ref = {}
            id_dict_ref_idx = {}
            for idx_ref, output in enumerate(outputs_ref):
                panoptic_seg_ids = torch.unique(output["panoptic_seg"][0])
                for panoptic_seg_id in panoptic_seg_ids:
                    # only consider thing_id
                    panoptic_class = panoptic_seg_id // 10000
                    if panoptic_class in metadata.thing_dataset_id_to_contiguous_id.values():
                        id_mask = output["panoptic_seg"][0] == int(panoptic_seg_id)
                        id_mask = id_mask > 0
                        if int(panoptic_seg_id) in id_dict_ref:
                            id_dict_ref[int(panoptic_seg_id)].append(id_mask)
                            id_dict_ref_idx[int(panoptic_seg_id)].append(idx_ref)
                        else:
                            id_dict_ref[int(panoptic_seg_id)] = [id_mask]
                            id_dict_ref_idx[int(panoptic_seg_id)] = [idx_ref]

            # integrate the id binary mask into one
            for main_key, main_val in id_dict.items():
                val_tensor = torch.stack(main_val, dim=0)
                id_dict[main_key] = torch.any(val_tensor, dim=0)
            for ref_key, ref_val in id_dict_ref.items():
                val_tensor = torch.stack(ref_val, dim=0)
                id_dict_ref[ref_key] = torch.any(val_tensor, dim=0)
                    
                
            # compare the ious between two dictionary
            # main to ref
            main_to_ref_dict = {}
            for main_key in id_dict.keys():
                for ref_key in id_dict_ref.keys():
                    main_to_ref = iou(id_dict[main_key], id_dict_ref[ref_key])
                    if main_to_ref > 0.4:
                        if main_key in main_to_ref_dict.keys():
                            main_to_ref_dict[main_key].append([ref_key, main_to_ref])
                        else:
                            main_to_ref_dict[main_key] = [[ref_key, main_to_ref]]
            main_to_ref_dict_match = {}
            for key in main_to_ref_dict:
                main_to_ref_dict_match[key] = sorted(main_to_ref_dict[key], key=lambda x: x[1], reverse=True)[0][0]

            ref_to_main_dict = {}
            for ref_key in id_dict_ref.keys():
                for main_key in id_dict.keys():
                    ref_to_main = iou(id_dict_ref[ref_key], id_dict[main_key])
                    if ref_to_main > 0.4:
                        if ref_key in ref_to_main_dict.keys():
                            ref_to_main_dict[ref_key].append([main_key, ref_to_main])
                        else:
                            ref_to_main_dict[ref_key] = [[main_key, ref_to_main]]
            ref_to_main_dict_match = {}
            for key in ref_to_main_dict:
                ref_to_main_dict_match[key] = sorted(ref_to_main_dict[key], key=lambda x: x[1], reverse=True)[0][0]

            def check_overlap(lists):
                set_lists = [set(lst) for lst in lists]
                common_elements = set.intersection(*set_lists)
                return len(common_elements) > 0

            # find the overlapping 
            main_to_ref_group = {}
            for main_id, ref_id in main_to_ref_dict_match.items():
                if ref_id in main_to_ref_group:
                    main_to_ref_group[ref_id].append(main_id)
                else:
                    main_to_ref_group[ref_id] = [main_id]
            main_to_ref_group_lists = list(main_to_ref_group.values())
            main_to_ref_group_lists_new = []
            for main_to_ref_group_ele in main_to_ref_group_lists:
                if len(main_to_ref_group_ele) > 1:
                    main_to_ref_group_lists_new.append(main_to_ref_group_ele)
            main_to_ref_group_lists_new_new = {}
            for main_to_ref_group_comp in main_to_ref_group_lists_new:
                check_list = []
                for main_to_ref in main_to_ref_group_comp:
                    check_list.append(id_dict_idx[main_to_ref])
                if not check_overlap(check_list):
                    main_to_ref_group_lists_new_new[main_to_ref_dict_match[main_to_ref_group_comp[0]]] = main_to_ref_group_comp

            ref_to_main_group = {}
            for ref_id, main_id in ref_to_main_dict_match.items():
                if main_id in ref_to_main_group:
                    ref_to_main_group[main_id].append(ref_id)
                else:
                    ref_to_main_group[main_id] = [ref_id]
            ref_to_main_group_lists = list(ref_to_main_group.values())
            ref_to_main_group_lists_new = []
            for ref_to_main_group_ele in ref_to_main_group_lists:
                if len(ref_to_main_group_ele) > 1:
                    ref_to_main_group_lists_new.append(ref_to_main_group_ele)
            ref_to_main_group_lists_new_new = {}
            for ref_to_main_group_comp in ref_to_main_group_lists_new:
                check_list = []
                for ref_to_main in ref_to_main_group_comp:
                    check_list.append(id_dict_ref_idx[ref_to_main])
                if not check_overlap(check_list):
                    ref_to_main_group_lists_new_new[ref_to_main_dict_match[ref_to_main_group_comp[0]]] = ref_to_main_group_comp
                    

            processed_results = []
            for (output, output_ref) in zip(outputs, outputs_ref):
                processed_results.append({})
                panoptic_seg = output["panoptic_seg"][0]
                panoptic_seg_ref = output_ref["panoptic_seg"][0]
                processed_results[-1]["original_image"] = output["original_image"]
                if len(main_to_ref_group_lists_new_new) > 0:
                    for ref_key, main_to_ref_group in main_to_ref_group_lists_new_new.items():
                        for ele in main_to_ref_group:
                            
                            panoptic_seg = torch.where(panoptic_seg==ele, main_to_ref_group[0], panoptic_seg.long()).type(torch.int32)
                # panoptic_seg_list.append(panoptic_seg)
                # if len(ref_to_main_group_lists_new_new) > 0:
                #     for main_key, ref_to_main_group in ref_to_main_group_lists_new_new.items():
                #         for ele in ref_to_main_group:
                #             panoptic_seg_ref = torch.where(panoptic_seg_ref==ele, ref_to_main_group[0], panoptic_seg_ref)
                        panoptic_seg = torch.where(panoptic_seg_ref==ref_key, main_to_ref_group[0], panoptic_seg.long()).type(torch.int32)
                    # import pdb; pdb.set_trace()    
                processed_results[-1]["panoptic_seg"] = [panoptic_seg, None]
            
            
            # for final_idx, (main_panop, ref_panop) in enumerate(zip(panoptic_seg_list, panoptic_seg_ref_list)):
            #     main_conf = outputs[final_idx]["panoptic_seg"][0][-1]
            #     ref_conf = outputs_ref[final_idx]["panoptic_seg"][0][-1]
            #     meta_thing = torch.Tensor(list(metadata.thing_dataset_id_to_contiguous_id.values())).to("cuda")
            #     main_stuff_mask = ~(torch.isin(main_panop//10000, meta_thing))
            #     meta_thing = torch.Tensor(list(metadata.thing_dataset_id_to_contiguous_id.values())).to("cuda")
            #     ref_stuff_mask = ~(torch.isin(ref_panop//10000, meta_thing))

            #     main_zero_mask = main_panop == 0
            #     ref_zero_mask = ref_panop == 0

            #     main_panop = torch.where(main_stuff_mask, (main_panop // 10000) * 10000 + 1000, main_panop)
            #     main_panop = torch.where(main_zero_mask, 0, main_panop)
            #     ref_panop = torch.where(main_stuff_mask, (ref_panop // 10000) * 10000 + 1000, ref_panop)
            #     ref_panop = torch.where(ref_zero_mask, 0, ref_panop)
            #     total_stuff_mask = main_stuff_mask * ref_stuff_mask

            #     main_panop_inst_id = main_panop % 10000
            #     new_panop = torch.where(
            #         main_conf > ref_conf, main_panop, ref_panop)

            #     real_new_panop = torch.where(total_stuff_mask, new_panop, main_panop)
            #     real_new_panop = torch.where(main_zero_mask, 0, real_new_panop)
            #     # real_new_panop = torch.where(ref_zero_mask, 0, real_new_panop)
            #     processed_results[final_idx]["panoptic_seg"] = [real_new_panop, None]

            # main_panop_dict = {}
            # main_panop_conf_dict = {}
            # for main_idx, main_panoptic in enumerate(panoptic_seg_list):
            #     main_ids = torch.unique(main_panoptic)
            #     for main_id in main_ids:
            #         main_id = int(main_id)
            #         if main_id == 0:
            #             continue
            #         mask = main_panoptic == main_id
            #         conf_score = torch.sum(mask * outputs[main_idx]["panoptic_seg"][0][-1]) / mask.sum()
            #         if main_id in main_panop_dict:
            #             main_panop_dict[main_id].append(mask)
            #             main_panop_conf_dict[main_id].append(conf_score)
            #         else:
            #             main_panop_dict[main_id] = [mask]
            #             main_panop_conf_dict[main_id] = [conf_score]
            # ref_panop_dict = {}
            # ref_panop_conf_dict = {}
            # for ref_idx, ref_panoptic in enumerate(panoptic_seg_ref_list):
            #     ref_ids = torch.unique(ref_panoptic)
            #     for ref_id in ref_ids:
            #         ref_id = int(ref_id)
            #         if ref_id == 0:
            #             continue
            #         mask = ref_panoptic == ref_id
            #         conf_score = torch.sum(mask * outputs_ref[ref_idx]["panoptic_seg"][0][-1]) / mask.sum()
            #         if ref_id in ref_panop_dict:
            #             ref_panop_dict[ref_id].append(mask)
            #             ref_panop_conf_dict[ref_id].append(conf_score)
            #         else:
            #             ref_panop_dict[ref_id] = [mask]
            #             ref_panop_conf_dict[ref_id] = [conf_score]


            # # integrate the id binary mask into one
            # for main_key, main_val in main_panop_dict.items():
            #     val_tensor = torch.stack(main_val, dim=0)
            #     main_panop_dict[main_key] = torch.any(val_tensor, dim=0)
            # for ref_key, ref_val in ref_panop_dict.items():
            #     val_tensor = torch.stack(ref_val, dim=0)
            #     ref_panop_dict[ref_key] = torch.any(val_tensor, dim=0)

            # # compare the ious between two dictionary
            # # main to ref
            # re_main_to_ref_dict = {}
            # for main_key in main_panop_dict.keys():
            #     for ref_key in ref_panop_dict.keys():
            #         main_to_ref = iou(main_panop_dict[main_key], ref_panop_dict[ref_key])
            #         if main_to_ref > 0.4:
            #             if main_key in re_main_to_ref_dict.keys():
            #                 re_main_to_ref_dict[main_key].append([ref_key, main_to_ref])
            #             else:
            #                 re_main_to_ref_dict[main_key] = [[ref_key, main_to_ref]]
            # re_main_to_ref_dict_match = {}
            # re_main_to_ref_dict_match_w_conf = {}
            # for key in re_main_to_ref_dict:
            #     re_main_to_ref_dict_match[key] = sorted(re_main_to_ref_dict[key], key=lambda x: x[1], reverse=True)[0][0]
            #     re_main_to_ref_dict_match_w_conf[key] = sorted(re_main_to_ref_dict[key], key=lambda x: x[1], reverse=True)[0][1]

            # # main to ref
            # re_ref_to_main_dict = {}
            # for ref_key in ref_panop_dict.keys():
            #     for main_key in main_panop_dict.keys():
            #         ref_to_main = iou(ref_panop_dict[ref_key], main_panop_dict[main_key])
            #         if ref_to_main > 0.4:
            #             if ref_key in re_ref_to_main_dict.keys():
            #                 re_ref_to_main_dict[ref_key].append([main_key, ref_to_main])
            #             else:
            #                 re_ref_to_main_dict[ref_key] = [[main_key, ref_to_main]]
            # re_ref_to_main_dict_match = {}
            # re_ref_to_main_dict_match_w_conf = {}
            # for key in re_ref_to_main_dict:
            #     re_ref_to_main_dict_match[key] = sorted(re_ref_to_main_dict[key], key=lambda x: x[1], reverse=True)[0][0]
            #     re_ref_to_main_dict_match_w_conf[key] = sorted(re_ref_to_main_dict[key], key=lambda x: x[1], reverse=True)[0][1]

            # double_matched_cur_to_prev = {}
            # for cur in re_main_to_ref_dict_match.keys():
            #     if re_main_to_ref_dict_match[cur] in re_ref_to_main_dict_match:
            #         if re_ref_to_main_dict_match[re_main_to_ref_dict_match[cur]] == cur:
            #             double_matched_cur_to_prev[cur] = [re_main_to_ref_dict_match[cur], re_main_to_ref_dict_match_w_conf[cur]]   

            # sorted_double_matched_cur_to_prev = sorted(double_matched_cur_to_prev.items(), key=lambda item: item[1][-1])

            # panoptic_seg_list_copy = copy.deepcopy(panoptic_seg_list)
            # for match_ids in sorted_double_matched_cur_to_prev:
            #     main_id = match_ids[0] 
            #     ref_id = match_ids[1][0]
            #     if main_id // 10000 == ref_id // 10000:
            #         new_id = main_id
            #     else:
            #         main_conf = re_main_to_ref_dict_match_w_conf[main_id]
            #         ref_conf = re_ref_to_main_dict_match_w_conf[ref_id]
            #         if main_conf > ref_conf:
            #             new_id = main_id
            #         else:
            #             new_id = main_id % 10000 + (ref_id // 10000) * 10000
            #     for final_idx, (output, output_ref) in enumerate(zip(panoptic_seg_list, panoptic_seg_ref_list)):
            #         ref_mask = output_ref == ref_id
            #         output = torch.where(ref_mask, new_id, output)
            #         panoptic_seg_list_copy[final_idx] = output

            # for pro_idx in range(len(processed_results)):
            #     processed_results[pro_idx]["panoptic_seg"] = [panoptic_seg_list_copy[pro_idx], None]

            # for main_id, ref_id in double_matched_cur_to_prev.items()



            # filter out the non-consistent areas
            # processed_results = []
            # for output, output_ref in zip(outputs, outputs_ref):
            #     processed_results.append({})
            #     processed_results[-1]["original_image"] = output["original_image"]
            #     cons_mask = (output["panoptic_seg"][0] // 10000) == (output_ref["panoptic_seg"][0] // 10000)
            #     processed_results[-1]["panoptic_seg"] = [torch.where(cons_mask, output["panoptic_seg"][0], 0), None]
            outputs = processed_results
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def iou(mask1, mask2, use_dice=True):
    # Convert masks to Boolean tensors
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    # Compute intersection and union
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()

    if use_dice:
        union = union + intersection
        intersection = intersection * 2

    # Handle special case when union is zero
    if union == 0:
        return 0.0

    # Compute IoU and return
    iou = intersection / union
    return iou.item()


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
