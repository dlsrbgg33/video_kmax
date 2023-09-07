# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_kmax_deeplab_config(cfg):
    """
    Add config for KMAX_DEEPLAB.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_panoptic_kmaxdeeplab"
    # Color augmentation
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.05
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.BACKBONE_FREEZE = False

    # kMaX-DeepLab model config
    cfg.MODEL.KMAX_DEEPLAB = CN()

    # whether to share matching results
    cfg.MODEL.KMAX_DEEPLAB.SHARE_FINAL_MATCHING = True

    # vis
    cfg.MODEL.KMAX_DEEPLAB.SAVE_VIS_NUM = 0

    cfg.MODEL.KMAX_DEEPLAB.USE_CUDNN = True

    # loss
    cfg.MODEL.KMAX_DEEPLAB.DEEP_SUPERVISION = True
    cfg.MODEL.KMAX_DEEPLAB.SKIP_CONN_INIT_VALUE = 0.0
    cfg.MODEL.KMAX_DEEPLAB.NO_OBJECT_WEIGHT = 1e-5
    cfg.MODEL.KMAX_DEEPLAB.CLASS_WEIGHT = 3.0
    cfg.MODEL.KMAX_DEEPLAB.DICE_WEIGHT = 3.0
    cfg.MODEL.KMAX_DEEPLAB.MASK_WEIGHT = 0.3
    cfg.MODEL.KMAX_DEEPLAB.INSDIS_WEIGHT = 1.0
    cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_WEIGHT = 1.0

    cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_TEMPERATURE = 1.5
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_SAMPLE_K = 4096
    cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_TEMPERATURE = 2.0
    cfg.MODEL.KMAX_DEEPLAB.UX_SEMANTIC_SAMPLE_K = 4096

    # Adding avg, 3 CA before kmax.
    cfg.MODEL.KMAX_DEEPLAB.ADVANCED_KMAX = False

    # pixel decoder config
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC = CN()
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.NAME = "kMaXPixelDecoder"
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.IN_FEATURES = ['res2', 'res3', 'res4', 'res5']
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.DEC_LAYERS = [1, 5, 1, 1]
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.LAYER_TYPES = ["axial", "axial", "bottleneck", "bottleneck"]
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.DEC_CHANNELS = [512, 256, 128, 64]
    cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.DROP_PATH_PROB = 0.0

    # transformer decoder config
    cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC = CN()
    cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME = "kMaXTransformerDecoder"
    cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.DEC_LAYERS = [2, 2, 2]
    cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NUM_OBJECT_QUERIES = 128
    cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.IN_CHANNELS = [2048, 1024, 512]
    cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.DROP_PATH_PROB = 0.0

    # kMaX-DeepLab inference config
    cfg.MODEL.KMAX_DEEPLAB.TEST = CN()
    cfg.MODEL.KMAX_DEEPLAB.TEST.SEMANTIC_ON = False
    cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON = False
    cfg.MODEL.KMAX_DEEPLAB.TEST.PANOPTIC_ON = True
    cfg.MODEL.KMAX_DEEPLAB.TEST.OBJECT_MASK_THRESHOLD = 0.4
    cfg.MODEL.KMAX_DEEPLAB.TEST.CLASS_THRESHOLD_THING = 0.7
    cfg.MODEL.KMAX_DEEPLAB.TEST.CLASS_THRESHOLD_STUFF = 0.5
    cfg.MODEL.KMAX_DEEPLAB.TEST.REORDER_CLASS_WEIGHT = 1.0
    cfg.MODEL.KMAX_DEEPLAB.TEST.REORDER_MASK_WEIGHT = 1.0
    cfg.MODEL.KMAX_DEEPLAB.TEST.THING_AREA_LIMIT = 0
    cfg.MODEL.KMAX_DEEPLAB.TEST.STUFF_AREA_LIMIT = 0
    cfg.MODEL.KMAX_DEEPLAB.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.KMAX_DEEPLAB.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.KMAX_DEEPLAB.SIZE_DIVISIBILITY = -1

    # https://github.com/SHI-Labs/OneFormer/blob/main/oneformer/config.py#L197
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.IN_CHANNELS = 3
    cfg.MODEL.CONVNEXT.DEPTHS = [3, 3, 27, 3]
    cfg.MODEL.CONVNEXT.DIMS = [192, 384, 768, 1536]
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE = 0.6
    cfg.MODEL.CONVNEXT.LSIT = 1e-6
    cfg.MODEL.CONVNEXT.OUT_INDICES = [0, 1, 2, 3]
    cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    cfg.INPUT.IMAGE_SIZE = [1281, 1281]
    cfg.INPUT.MIN_SCALE = 0.3
    cfg.INPUT.MAX_SCALE = 1.7
    
    cfg.FIND_UNUSED_PARAMETERS = False

    # For video-kMaX
    
    # Video Sampler
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 5 # set 5 as a default frame range for VPS
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.TEST_SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.DEBUG_ONLY = False

    cfg.INPUT.WANDB_VISUALIZE = True

    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
    # Video-kMaX model config
    cfg.MODEL.VIDEO_KMAX = CN()

    # Video-kMaX inference config
    cfg.MODEL.VIDEO_KMAX.TEST = CN()
    cfg.MODEL.VIDEO_KMAX.TEST.USE_CLIP_STITCHING = False
    cfg.MODEL.VIDEO_KMAX.TEST.MEMORY_NAME = 'Hi-LAMB' # ['None', 'IDOL', 'Hi-LAMB']
    cfg.MODEL.VIDEO_KMAX.TEST.LAMB_CONF_THRES = 0.7 # ['IDOL', 'Hi-LAMB']
    cfg.MODEL.VIDEO_KMAX.TEST.LAMB_BUFFER_SIZE = 10 # ['IDOL', 'Hi-LAMB']
    # cfg.MODEL.VIDEO_KMAX.TEST.MEMORY_NAME = None # ['None', 'IDOL', 'Hi-LAMB']
    cfg.MODEL.VIDEO_KMAX.TEST.AGGREGATE_METHOD = 'IDOL' # ['IDOL', 'Hi-LAMB']
    cfg.MODEL.VIDEO_KMAX.TEST.SEMANTIC_CHECK = True
    
    cfg.MODEL.VIDEO_KMAX.SAM_FUSE = False

    cfg.WANDB = CN()
    cfg.WANDB.IS_AVAILBLE = True
    cfg.WANDB.USE_VISUALIZER = True
    cfg.WANDB.NAME = ['vkmax', '0509_convnext_xxl_backfreeze_50k_coco'] # [project_name, exp_name]
    