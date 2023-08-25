import torch
from torch import nn
from torch.nn import functional as F
import copy
import numpy as np

class VideoStitching(nn.Module):
    def __init__(self, cfg, num_frames, metadata):
        super().__init__()

        self._num_frames = num_frames
        self.label_divisor = metadata.label_divisor
        self.stitching_with_semantic_id_checked = True

        self.num_mask_slot = cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NUM_OBJECT_QUERIES

        # self._num_stitching_overlap_frames = self._num_frames - 1
        # if near-online, num_stitching_overlap=1, otherwise num_stitching_overlap=0
        # hard-coded for near-online
        if num_frames > 1:
            # Near-online
            self._num_stitching_overlap_frames = 1
        elif num_frames == 1:
            # Online
            self._num_stitching_overlap_frames = 0
        else:
            raise ValueError('num_frames must be greater or equal to 1.')      
        

        self._previous_clip_prediction_overlap = torch.zeros(0)
        self.id_overlap_offset = self.label_divisor // 2 -1
        self.use_semantic_check = cfg.MODEL.VIDEO_KMAX.TEST.SEMANTIC_CHECK

        self.next_tracking_id = 0

    def _intra_clip_associate(self, panoptic_seg, aux_ids_feat, seq_idx, height):
        panoptic_seg_split = torch.split(panoptic_seg, height, dim=0)
        
        first_frame_thing_ids = []
        first_frame_total_ids = panoptic_seg_split[0].unique()
        for first_id in first_frame_total_ids:
            if int(first_id) in aux_ids_feat.keys():
                first_frame_thing_ids.append(int(first_id))

        remain_frame_thing_ids = []
        remain_frame_total_ids = torch.cat(panoptic_seg_split[1:], dim=0).unique()
        for remain_id in remain_frame_total_ids:
            if int(remain_id) in aux_ids_feat.keys():
                remain_frame_thing_ids.append(int(remain_id))
        
        # check if there is any overlap
        first_not_overlap_thing_ids = []
        for first_thing_id in first_frame_thing_ids:
            if first_thing_id not in remain_frame_thing_ids:
                first_not_overlap_thing_ids.append(first_thing_id)

        reamin_not_overlap_thing_ids = []
        for remain_thing_id in remain_frame_thing_ids:
            if remain_thing_id not in first_frame_thing_ids:
                reamin_not_overlap_thing_ids.append(remain_thing_id)

        first_not_overlap_length = len(first_not_overlap_thing_ids)
        remain_not_overlap_length = len(reamin_not_overlap_thing_ids)

        if first_not_overlap_length == 0 or remain_not_overlap_length == 0:
            return panoptic_seg, aux_ids_feat

        # get related feat from first_not_overlap_thing_ids
        first_aux_feat_app = []
        first_aux_feat_loc = []
        for first_not_over in first_not_overlap_thing_ids:
            first_aux_feat_app.append(aux_ids_feat[first_not_over]["cluster_feat"])
            first_aux_feat_loc.append(aux_ids_feat[first_not_over]["bbox_xyxy"])
        first_aux_feat_app = torch.stack(first_aux_feat_app, dim=0)
        first_aux_feat_loc = torch.stack(first_aux_feat_loc, dim=0)
        
        # get related feat from reamin_not_overlap_thing_ids
        remain_aux_feat_app = []
        remain_aux_feat_loc = []
        for remain_not_over in reamin_not_overlap_thing_ids:
            remain_aux_feat_app.append(aux_ids_feat[remain_not_over]["cluster_feat"])
            remain_aux_feat_loc.append(aux_ids_feat[remain_not_over]["bbox_xyxy"])
        remain_aux_feat_app = torch.stack(remain_aux_feat_app, dim=0)
        remain_aux_feat_loc = torch.stack(remain_aux_feat_loc, dim=0)

        appearance_scores = torch.mm(
                    F.normalize(first_aux_feat_app, p=2, dim=1),
                    F.normalize(remain_aux_feat_app, p=2, dim=1).t())
        location_gap = torch.cdist(first_aux_feat_loc, remain_aux_feat_loc, p=2)
        location_scores = torch.exp(-location_gap/2.0)
        # import pdb
        # pdb.set_trace()
        total_score = appearance_scores * location_scores   
        
        from scipy.optimize import linear_sum_assignment

        # if seq_idx == 8:
        #     import pdb; pdb.set_trace()

        total_score_np = np.array(total_score.cpu())
        row_ind, col_ind = linear_sum_assignment(-total_score_np)

        re_stitched_aux_dict = {}
        for idx in range(len(row_ind)):
            current_corres_id = first_not_overlap_thing_ids[row_ind[idx]]
            memory_corres_id = reamin_not_overlap_thing_ids[col_ind[idx]]
            score = total_score[row_ind[idx], col_ind[idx]]

            if float(score) > 0.5: 
                binary_mask = (panoptic_seg == memory_corres_id)
                panoptic_seg[binary_mask] = current_corres_id
                
                aux_ids_feat[current_corres_id]["cluster_feat"] = (
                    aux_ids_feat[current_corres_id]["cluster_feat"] + aux_ids_feat[memory_corres_id]["cluster_feat"]) / 2.0
                aux_ids_feat[current_corres_id]["bbox_xyxy"] = (
                    aux_ids_feat[current_corres_id]["bbox_xyxy"] + aux_ids_feat[memory_corres_id]["bbox_xyxy"]) / 2.0
                
                # pop out memory_corres_id from aux_ids_feat
                del aux_ids_feat[memory_corres_id]
        return panoptic_seg, aux_ids_feat

            

    def forward(self, panoptic_seg, aux_ids_feat, seq_idx, height):

        # panoptic_seg: [H, W], each represents "id" + label_divisor * "category_id"
        # all stuff and things start from 1, 0 means "void"

        # check intra-clip id difference
        # if self._num_stitching_overlap_frames != 0:
        #     panoptic_seg, aux_ids_feat = self._intra_clip_associate(panoptic_seg, aux_ids_feat, seq_idx, height)

        stitched_panoptic, non_match_dict, matched_cur_to_prev = self._get_and_update_stitched_video_predictions(panoptic_seg, seq_idx)

        # For each time step, we only need to save one frame of the clip
        # (eval_frame_index) for evaluation, since the IDs in all overlapping and
        # non-overlapping predictions are matched with (or propagated from) the
        # predictions in previous time step.
        
        self.save_stitched_panoptic_to_buffer(stitched_panoptic)
                

        # num_frames_height, width = stitched_panoptic.shape
        # height = num_frames_height // self._num_frames
        # splited_stitched_panoptic = stitched_panoptic[:height * (
        #             self._num_frames - self._num_stitching_overlap_frames)]

        return stitched_panoptic, non_match_dict, matched_cur_to_prev, aux_ids_feat

    def _get_and_update_stitched_video_predictions(self, panoptic_seg, seq_idx):
        """Returns stitched video predictions and updates results."""

        # prev_panoptic_overlap is the last num_stitching_overlap_frames predictions
        # from the previous time step.
        if self._num_stitching_overlap_frames != 0:
            prev_panoptic_overlap = self._get_previous_clip_prediction_overlap(
                panoptic_seg, seq_idx)

        if seq_idx == 0:
            self._clean_storage_every_seq(panoptic_seg)
            return panoptic_seg, None, None
        else:
            # The segment matching is done among the tracklet tubes.
            current_panoptic = panoptic_seg
            
            if self._num_stitching_overlap_frames != 0:
                stitched_panoptic, non_match_dict, matched_cur_to_prev = (
                    self._stitch_video_panoptic_prediction(
                        prev_panoptic_overlap,
                        current_panoptic, seq_idx))
            else:
                # TODO) need to correct for online
                stitched_panoptic = self._offset_added(current_panoptic)
                stitched_panoptic, non_match_dict = self._get_and_assign_consecutive_next_tracking_id(
                    stitched_panoptic)
                matched_cur_to_prev = None
            return stitched_panoptic, non_match_dict, matched_cur_to_prev


    def _get_previous_clip_prediction_overlap(self, panoptic_seg, seq_idx):
        """Gets num_stitching_overlap_frames prediction from previous time step.

        Returns previous prediction for video stitching. It is saved to a buffer if
        the first frame of a sequence is being processed, or retrieved from the
        buffer otherwise.

        Args:
          pred_panoptic: A torch.Tensor of shape [num_frames * height,  width].
          seq_idx: A torch.Tensor of shape [1], indicating the input sequence ID.

        Returns:
          prev_panopic_overlap: A torch.Tensor of shape [batch, height,
            num_stitching_overlap_frames * width], which is the last
            num_stitching_overlap_frames of the previous panoptic prediction
             sequence.
        """
        pred_panoptic = panoptic_seg
        num_frames_height, width = pred_panoptic.shape
        height = num_frames_height // self._num_frames # ex) 720 // 3 = 240
        
        def _store_first_frame_prediction():
            # splited_pred_panoptic -> should be the last frame of the clip
            splited_pred_panoptic = copy.deepcopy(pred_panoptic[height * (
                    self._num_frames - self._num_stitching_overlap_frames):])
            self._previous_clip_prediction_overlap = splited_pred_panoptic
            return splited_pred_panoptic

        def _load_previous_prediction_overlap():
            prev_panoptic_overlap = self._previous_clip_prediction_overlap.clone()
            return prev_panoptic_overlap


        prev_panoptic_overlap = (_store_first_frame_prediction() if seq_idx == 0
            else _load_previous_prediction_overlap())
        prev_panoptic_overlap = prev_panoptic_overlap.view(
            [height * self._num_stitching_overlap_frames, width])
        return prev_panoptic_overlap


    def save_stitched_panoptic_to_buffer(self, stitched_panoptic):

        num_frames_height, width = stitched_panoptic.shape
        height = num_frames_height // self._num_frames

        split_stitched_panoptic = stitched_panoptic[height * (
            self._num_frames - self._num_stitching_overlap_frames):]
        self._previous_clip_prediction_overlap = copy.deepcopy(split_stitched_panoptic)
        
        max_stitched_panoptic = int((stitched_panoptic.unique() % self.label_divisor).max())
        # select the max value between the max_stitched_panoptic and the next_tracking_id
        self.next_tracking_id = max_stitched_panoptic

    def _clean_storage_every_seq(self, panoptic_seg):
        self._previous_clip_prediction_overlap = torch.zeros(0)
        max_id = int((panoptic_seg.unique() % self.label_divisor).max())
        # select the max value between the max_stitched_panoptic and the next_tracking_id
        self.next_tracking_id = max_id

    def _stitch_video_panoptic_prediction(self,
                                          prev_panoptic_overlap,
                                          current_panoptic, seq_id):
        height, width = prev_panoptic_overlap.shape

        current_panoptic = self._offset_added(current_panoptic)
        
        current_panoptic_copy = copy.deepcopy(current_panoptic)
        # current overlap should be the first frame of the clip
        current_panoptic_overlap = current_panoptic[:height]
        # Bi-directional matching
        match_ids_cur_to_prev = self._compute_and_sort_iou_between_panoptic_ids(
            prev_panoptic_overlap, current_panoptic_overlap
        ) 
        match_ids_prev_to_cur = self._compute_and_sort_iou_between_panoptic_ids(
            current_panoptic_overlap, prev_panoptic_overlap
        ) 
        double_matched_cur_to_prev = {}
        for cur in match_ids_cur_to_prev.keys():
            if match_ids_cur_to_prev[cur] in match_ids_prev_to_cur:
                if match_ids_prev_to_cur[match_ids_cur_to_prev[cur]] == cur:
                    double_matched_cur_to_prev[cur] = match_ids_cur_to_prev[cur]        
 
        # check whether match ids have the same semantic id
        if self.use_semantic_check:
            double_matched_cur_to_prev_cons = self._semantic_id_consistency_checked(double_matched_cur_to_prev)
        else:
            double_matched_cur_to_prev_cons = double_matched_cur_to_prev

        stitched_panoptic = self._match_and_propagate_ids(
            current_panoptic_copy, double_matched_cur_to_prev_cons)

        stitched_panoptic, unmatched_dict = self._get_and_assign_consecutive_next_tracking_id(stitched_panoptic)
        
        return stitched_panoptic, unmatched_dict, double_matched_cur_to_prev_cons
    
    def _offset_added(self, current_panoptic):
        adding_binary_mask = (current_panoptic != 0) * self.id_overlap_offset
        current_panoptic += adding_binary_mask
        
        return current_panoptic
    
    def _get_and_assign_consecutive_next_tracking_id(self, stitched_panoptic):
        unique_panoptic_id = torch.unique(stitched_panoptic)
        
        unique_id = unique_panoptic_id % self.label_divisor
        # current_max_id_before_offset = prev_unique_id.max()
        
        current_max_id_before_offset = self.next_tracking_id
        after_offset = unique_panoptic_id[unique_id > self.id_overlap_offset]
        after_offset_sorted = torch.sort(after_offset)[0]
        unmatched_dict = {}
        for id_after_offset in after_offset_sorted:
            current_max_id_before_offset += 1
            category_id = id_after_offset // self.label_divisor
            binary_mask = stitched_panoptic == id_after_offset
            current_convert_mask = binary_mask * current_max_id_before_offset + category_id * self.label_divisor
            unmatched_dict[int(id_after_offset)] = int(current_max_id_before_offset + category_id * self.label_divisor)
            stitched_panoptic = torch.where(binary_mask, current_convert_mask.long(), stitched_panoptic.long())

        self.next_tracking_id = current_max_id_before_offset

        stitched_panoptic = stitched_panoptic.type(torch.int)
        return stitched_panoptic, unmatched_dict
    
    def _match_and_propagate_ids(self, current_panoptic, double_matched_cur_to_prev):
        for cur in double_matched_cur_to_prev.keys():
            binary_mask = current_panoptic == cur 
            replaced_id = double_matched_cur_to_prev[cur]
            replaced_id_mask = binary_mask * replaced_id
            current_panoptic = torch.where(binary_mask, replaced_id_mask.long(), current_panoptic.long())
        current_panoptic = current_panoptic.type(torch.int)


        return current_panoptic

    def _semantic_id_consistency_checked(self, double_matched_cur_to_prev):
        double_matched_cur_to_prev_copy = copy.deepcopy(double_matched_cur_to_prev)
        for cur_id in list(double_matched_cur_to_prev.keys()):
            cur_category = cur_id // self.label_divisor
            prev_category = double_matched_cur_to_prev[cur_id] // self.label_divisor
            if cur_category != prev_category:
                # pop out the unmatched ids
                double_matched_cur_to_prev_copy.pop(cur_id)
        return double_matched_cur_to_prev_copy
            
    def _compute_and_sort_iou_between_panoptic_ids(self, pred_panop_1, pred_panop_2):

        # Compute IoU for each pair of objects
        ious = {}
        for id1 in torch.unique(pred_panop_1):
            if id1 == 0:
                continue
            for id2 in torch.unique(pred_panop_2):
                if id2 == 0:
                    continue
                iou_value = self._iou(pred_panop_1 == id1, pred_panop_2 == id2, use_dice=True)
                if int(id2) in ious:
                    ious[int(id2)].append((int(id1), iou_value))
                else:
                    ious[int(id2)] = [(int(id1), iou_value)]

        for key in ious:
            ious[key] = sorted(ious[key], key=lambda x: x[1], reverse=True)
        
        total_sorted_keys = sorted(ious.keys(), key=lambda x: ious[x][0][1], reverse=True)

        new_matching_ids = {}
        matched_ids = []
        for key in total_sorted_keys:
            for idx in range(len(ious[key])):
                if ious[key][idx][0] not in matched_ids:
                    new_matching_ids[key] = ious[key][idx][0]
                    matched_ids.append(ious[key][idx][0])
                    break
        return new_matching_ids
    
    def _iou(self, mask1, mask2, use_dice=False):
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

    def update_aux_ids(self, aux_dict, non_match_dict):
        # aux_dict has "class_conf", "cluster_feat" or "bbox_xyxy"
        if non_match_dict is None:
            return non_match_dict
        new_dict = {}
        aux_dict = copy.deepcopy(aux_dict)
        for id_bef_stitch, id_after_stitch in non_match_dict.items():
            id_bef_stitch = int(id_bef_stitch) - self.id_overlap_offset
            new_dict[int(id_after_stitch)] = aux_dict[id_bef_stitch]
        return new_dict