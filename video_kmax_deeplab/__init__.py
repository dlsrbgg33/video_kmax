from . import data  # register all new datasets
from . import modeling

# config
from .config import add_kmax_deeplab_config

# dataset loading
from .data.dataset_mappers.coco_panoptic_kmaxdeeplab_dataset_mapper import COCOPanoptickMaXDeepLabDatasetMapper
from .data.dataset_mappers.vip_seg_panoptic_dataset_mapper import VIPSegPanopticDatasetMapper
from .data.dataset_mappers.kittistep_panoptic_dataset_mapper import KITTIPanopticDatasetMapper
from .data.dataset_mappers.coco_video_panoptic_dataset_mapper import COCOVideoPanopticDatasetMapper

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

