from . import data  # register all new datasets
from . import modeling

# config
from .config import add_kmax_deeplab_config

# dataset loading
from .data.dataset_mappers.coco_panoptic_kmaxdeeplab_dataset_mapper import COCOPanoptickMaXDeepLabDatasetMapper
from .data.dataset_mappers.vip_seg_panoptic_dataset_mapper import VIPSegPanopticDatasetMapper
from .data.dataset_mappers.coco_video_panoptic_dataset_mapper import COCOVideoPanopticDatasetMapper
from .data.dataset_mappers.burst_video_sem_dataset_mapper import BURSTVideoDatasetMapper
from .data.dataset_mappers.ytvis_instance_kmaxdeeplab_dataset_mapper import InstancekMaXDeepLabDatasetMapper

from .data.dataset_mappers.ytbvis_19_mask_dataset_mapper import YTVISDatasetMapper

# video
from .data import (
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
# models
# from .bert_model import BertEncoder
from .kmax_model import kMaXDeepLab
from .video_kmax_model import VideokMaX
from .video_kmax_model_instance import VideokMaXInst

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
