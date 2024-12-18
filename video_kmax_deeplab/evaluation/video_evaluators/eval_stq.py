# -------------------------------------------------------------------
# Video Panoptic Segmentation
#
# VPQ evaluation code by tube (video segment) matching
# Inference on every frames and evaluation on every 5 frames.
# ------------------------------------------------------------------

import argparse
import sys
import os
import os.path
import numpy as np
from PIL import Image
import multiprocessing
import time
import json
from tqdm import tqdm
from collections import defaultdict
import copy
import pdb
import video_kmax_deeplab.evaluation.video_evaluators.segmentation_and_tracking_quality as numpy_stq

from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='VPSNet eval')
    parser.add_argument('--submit_dir', '-i', type=str,
                        help='test output directory')

    parser.add_argument('--truth_dir', type=str,
                        help='ground truth directory. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panomasksRGB '
                             'after running the conversion script')

    parser.add_argument('--pan_gt_json_file', type=str,
                        help='ground truth JSON file. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panoptic_gt_'
                             'VIPSeg_val.json after running the conversion script')

    args = parser.parse_args()
    return args


def STQeval(predictions, pan_gt_json_file, pred_dir, gt_folder, output_dir):
    n_classes = 124 # need to modify to be corresponding to each dataset 
    ignore_label = 255
    bit_shit = 16
    
    start_all = time.time()

    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)

    categories = gt_jsons['categories']

    thing_list_ = []
    for cate_ in categories:
        cat_id = cate_['id']
        isthing = cate_['isthing']
        if isthing:
            thing_list_.append(cat_id)

    stq_metric = numpy_stq.STQuality(n_classes, thing_list_, ignore_label,
                                     bit_shit, 2**24)

    pred_annos = predictions
    pred_j={}
    for p_a in pred_annos:
        pred_j[p_a['video_id']] = p_a['annotations']
    gt_annos = gt_jsons['annotations']
    gt_j  ={}
    for g_a in gt_annos:
        gt_j[g_a['video_id']] = g_a['annotations']
     

    gt_pred_split = []

    pbar = tqdm(gt_jsons['videos'])
    for seq_id, video_images in enumerate(pbar):
        video_id = video_images['video_id']
        pbar.set_description(video_id)

        # print('processing video:{}'.format(video_id))
        gt_image_jsons = video_images['images']
        gt_js = gt_j[video_id]
        pred_js = pred_j[video_id]
#    gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
        assert len(gt_js) == len(pred_js)
#    nframes_per_video = 6
#    vid_num = len(gt_jsons)//nframes_per_video # 600//6 = 100
        gt_pans =[]
        pred_pans = []
        for imgname_j in gt_image_jsons:
            imgname = imgname_j['file_name']
            image = np.array(Image.open((os.path.join(pred_dir, video_id, imgname))))
            pred_pans.append(image)
            image = np.array(Image.open(os.path.join(gt_folder, video_id, imgname)))
            gt_pans.append(image)
        gt_id_to_ins_num_dic={}
        list_tmp = []
        for segm in gt_js:
            for img_info in segm['segments_info']:
                id_tmp_ = img_info['id']
                if id_tmp_ not in list_tmp:
                    list_tmp.append(id_tmp_)
        for ii, id_tmp_ in enumerate(list_tmp):
            gt_id_to_ins_num_dic[id_tmp_]=ii
            
        pred_id_to_ins_num_dic={}
        list_tmp = []
        for segm in pred_js:
            for img_info in segm['segments_info']:
                id_tmp_ = img_info['id']
                if id_tmp_ not in list_tmp:
                    list_tmp.append(id_tmp_)
        for ii, id_tmp_ in enumerate(list_tmp):
            pred_id_to_ins_num_dic[id_tmp_]=ii

        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(list(zip(gt_js,pred_js,gt_pans,pred_pans,gt_image_jsons))):
            #### Step1. Collect frame-level pan_gt, pan_pred, etc.
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

            ground_truth_instance = np.ones_like(pan_gt)*255
            ground_truth_semantic = np.ones_like(pan_gt)*255
            for el in gt_json['segments_info']:
                id_ = el['id']
                cate_id = el['category_id']
                ground_truth_semantic[pan_gt==id_] = cate_id
                ground_truth_instance[pan_gt==id_] = gt_id_to_ins_num_dic[id_]

            ground_truth = ((ground_truth_semantic << bit_shit) + ground_truth_instance)

            prediction_instance = np.ones_like(pan_pred)*255
            prediction_semantic = np.ones_like(pan_pred)*255

            for el in pred_json['segments_info']:
                id_ = el['id']
                cate_id = el['category_id']
                prediction_semantic[pan_pred==id_] = cate_id
                prediction_instance[pan_pred==id_] = pred_id_to_ins_num_dic[id_]
            prediction = ((prediction_semantic << bit_shit) + prediction_instance)  

            stq_metric.update_state(ground_truth.astype(dtype=np.int32),
                              prediction.astype(dtype=np.int32), seq_id) 
    result = stq_metric.result()         

    output_filename = os.path.join(output_dir, 'stq-final.txt')
    output_file = open(output_filename, 'w')
    output_file.write("STQ:%.4f\n"%(result['STQ']))
    output_file.write("SQ:%.4f\n"%(result['IoU']))
    output_file.write("AQ:%.4f\n"%(result['AQ']))
    output_file.close()
    print('==> STQ time All:', time.time() - start_all, 'sec')

    print('*'*100)
    print('STQ : {}'.format(result['STQ']))
    print('AQ :{}'.format(result['AQ']) )
    print('IoU:{}'.format(result['IoU']))
    print('STQ_per_seq')
    print(result['STQ_per_seq'])
    print('AQ_per_seq')
    print(result['AQ_per_seq'])
    print('ID_per_seq')
    print(result['ID_per_seq'])
    print('Length_per_seq')
    print(result['Length_per_seq'])
    print('*'*100)

    results = OrderedDict({"video_panoptic_seg": result})

    return results
