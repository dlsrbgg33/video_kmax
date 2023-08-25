# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from typing import Dict, Optional, Union
import torch
from fvcore.common.history_buffer import HistoryBuffer

from detectron2.config import CfgNode
from detectron2.utils.file_io import PathManager

from detectron2.utils.events import EventWriter

from detectron2.utils.events import get_event_storage

__all__ = [
    "WandbWriter",
]


class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    """

    def __init__(
        self,
        project: list = ["detectron2", "date"],
        config: Union[Dict, CfgNode] = {},  # noqa: B006
        window_size: int = 20,
        **kwargs,
    ):
        """
        Args:
            project (str): W&B Project name
            config Union[Dict, CfgNode]: the project level configuration object
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `wandb.init(...)`
        """
        import wandb

        self._window_size = window_size
        self._run = wandb.run
        self._run._label(repo="detectron2")

    def write(self):
        storage = get_event_storage()
        log_dict = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v

        self._run.log(log_dict)

    def close(self):
        self._run.finish()