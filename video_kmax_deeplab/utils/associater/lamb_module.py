# class name as "LAMB", memory buffer to store previous ids and decode

import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

class LAMB(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()

        self.score_thres = cfg.MODEL.VIDEO_KMAX.TEST.LAMB_CONF_THRES
        self.buffer_size = cfg.MODEL.VIDEO_KMAX.TEST.LAMB_BUFFER_SIZE

        self.metadata = metadata
        self.label_divisor = metadata.label_divisor

        self.temperature = cfg.MODEL.VIDEO_KMAX.TEST.LAMB_TEMPERATURE

    def _encode(self, aux_ids_feat, memory_dict, current_clip):

        for aux_key in aux_ids_feat.keys():
            if aux_key in memory_dict:
                prev_feature = memory_dict[aux_key]["cluster_feat"]
                new_feature = aux_ids_feat[aux_key]["cluster_feat"]
                # currently fixed alpha as 0.8
                memory_dict[aux_key]["cluster_feat"] = 0.8 * prev_feature + 0.2 * new_feature
                memory_dict[aux_key]["bbox_xyxy"].append(aux_ids_feat[aux_key]["bbox_xyxy"])
                memory_dict[aux_key]["bbox_xyxy"] = memory_dict[aux_key]["bbox_xyxy"][-2:]
            else:
                memory_dict[aux_key] = aux_ids_feat[aux_key]
                # copy same bbox to calculate the velocity
                memory_dict[aux_key]["bbox_xyxy"] = [aux_ids_feat[aux_key]["bbox_xyxy"],
                                                     aux_ids_feat[aux_key]["bbox_xyxy"]]
            memory_dict[aux_key]["buffer_frame"] = current_clip

        for mem_key in list(memory_dict.keys()):
            if mem_key not in aux_ids_feat:
                pprev_bbox = memory_dict[mem_key]["bbox_xyxy"][-2] 
                prev_bbox = memory_dict[mem_key]["bbox_xyxy"][-1] 
                aug_bbox = prev_bbox + prev_bbox - pprev_bbox
                # below is the assumption that bbox is not out of the image
                # if aug_bbox.min()>=0 and aug_bbox.max()<=1:
                memory_dict[mem_key]["bbox_xyxy"].append(aug_bbox)
                memory_dict[mem_key]["bbox_xyxy"] = memory_dict[mem_key]["bbox_xyxy"][-2:]
            if "buffer_frame" in memory_dict[mem_key]:
                if current_clip - memory_dict[mem_key]["buffer_frame"] > self.buffer_size: 
                    memory_dict.pop(mem_key)


    def remove_stuff(self, list_of_dict):
        for dict_to_list in list_of_dict:
            if dict_to_list is not None:
                dict_to_list_copy = copy.deepcopy(dict_to_list)
                for k, v in dict_to_list_copy.items():
                    category = k // self.label_divisor
                    if int(category) not in self.metadata.thing_dataset_id_to_contiguous_id.values():
                        del dict_to_list[k]


    def _remove_matched(self, memory_dict, matched_cur_to_prev):
        matched_ids = matched_cur_to_prev.values()
        memory_dict_copy = copy.deepcopy(memory_dict)
        for matched_id in matched_ids:
            if matched_id in memory_dict_copy:
                memory_dict.pop(matched_id)
                
                
    def _find_matched_and_restitch(self, stitched_panop_seg,
                                   non_stitched_aux_dict, memory_dict, current_clip):

        memory_dict_corres_ids = [x for x in memory_dict.keys()]
        memory_dict_app_feat = torch.stack(
            [x["cluster_feat"] for x in memory_dict.values()], dim=0)
        memory_dict_loc_feat = torch.stack(
            [x["bbox_xyxy"][-1] for x in memory_dict.values()], dim=0)

        current_dict_corres_ids = [x for x in non_stitched_aux_dict.keys()]
        current_dict_app_feat = torch.stack(
            [x["cluster_feat"] for x in non_stitched_aux_dict.values()], dim=0)
        current_dict_loc_feat = torch.stack(
            [x["bbox_xyxy"] for x in non_stitched_aux_dict.values()], dim=0)

        appearance_scores = torch.mm(
                    F.normalize(current_dict_app_feat, p=2, dim=1),
                    F.normalize(memory_dict_app_feat, p=2, dim=1).t())
        location_gap = torch.cdist(current_dict_loc_feat, memory_dict_loc_feat, p=2)
        location_scores = torch.exp(-location_gap/self.temperature)

        total_score = appearance_scores * location_scores
        
        total_score_np = np.array(total_score.cpu())
        row_ind, col_ind = linear_sum_assignment(-total_score_np)

        re_stitched_aux_dict = {}
        for idx in range(len(row_ind)):
            current_corres_id = current_dict_corres_ids[row_ind[idx]]
            memory_corres_id = memory_dict_corres_ids[col_ind[idx]]
            cur_class, mem_class = (
                current_corres_id // self.label_divisor, memory_corres_id // self.label_divisor)
            score = total_score[row_ind[idx], col_ind[idx]]

            if float(score) > self.score_thres and cur_class == mem_class:
                re_stitched_aux_dict[memory_corres_id] = non_stitched_aux_dict[current_corres_id]
                binary_mask = (stitched_panop_seg == current_corres_id)
                stitched_panop_seg[binary_mask] = memory_corres_id

        return stitched_panop_seg, re_stitched_aux_dict

    def _decode(self, stitched_panop_seg, non_stitched_aux_dict,
               matched_cur_to_prev, memory_dict, current_clip):
        # compare between non_stitched_aux_dict and memory_dict
        # convert the all dict (non_stitch + re-stitch) 
        # convert stithced_panop_seg

        # 1. remove matched ones from memory_dict
        memory_dict_copy = copy.deepcopy(memory_dict)
        if matched_cur_to_prev is not None:
            self._remove_matched(memory_dict_copy, matched_cur_to_prev)
        
        # 2. compare non_stitched_aux_dict and memory_dict_copy
        # 2.1 find the matched ones
        # if all of the ids in memory are already matched, skip decoding.
        if len(memory_dict_copy) > 0:
            restithced_panop_seg, re_stitched_aux_dict = self._find_matched_and_restitch(
                stitched_panop_seg, non_stitched_aux_dict, memory_dict_copy, current_clip)
        else:
            restithced_panop_seg, re_stitched_aux_dict = stitched_panop_seg, non_stitched_aux_dict

        return restithced_panop_seg, re_stitched_aux_dict
        

    def convert_aux_ids(self, aux_ids, matched_cur_to_prev):
        new_aux_dict = {}
        for aux_key, aux_val in aux_ids.items():
            aux_key_offset = aux_key + self.label_divisor // 2 - 1
            if aux_key_offset in matched_cur_to_prev:
                new_aux_dict[matched_cur_to_prev[aux_key_offset]] = aux_val

        return new_aux_dict


    def memory_decode_and_encode(self,
                                 prev_stitched_seg,
                                 non_stitched_ids_feat,
                                 stitched_ids_feat,
                                 aux_ids_feat,
                                 matched_cur_to_prev,
                                 memory_dict,
                                 current_clip):

        if non_stitched_ids_feat is not None:
            # first decoding
            if len(non_stitched_ids_feat) > 0:
                cur_stitched_seg, re_stitched_aux_dict = self._decode(prev_stitched_seg, non_stitched_ids_feat,
                                                                        matched_cur_to_prev, memory_dict, current_clip)
                # all_dict = {} / add non_stitched_dict and stitched_aux_dict into all_dict
                aux_ids_feat = {}
                aux_ids_feat.update(re_stitched_aux_dict)
                if stitched_ids_feat is not None:
                    aux_ids_feat.update(stitched_ids_feat)
            else:
                # aux_ids_feat should be changed here
                aux_ids_feat = self.convert_aux_ids(aux_ids_feat,
                            matched_cur_to_prev)
                cur_stitched_seg = prev_stitched_seg 
        else:
            cur_stitched_seg = prev_stitched_seg
        self._encode(aux_ids_feat, memory_dict, current_clip)
        return cur_stitched_seg
