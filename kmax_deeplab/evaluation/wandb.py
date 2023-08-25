import itertools

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator


class WandbVideoVisualizer(DatasetEvaluator):
    """
    WandbVisualizer logs the model predictions over time in form of interactive W&B Tables.
    Sopported tasks: Bounding Box Detection, Semantic Segmentation
    """

    def __init__(self, dataset_name, wandb_name, size=50) -> None:
        """
        Args:
            dataset_name (str): Name of the registered dataset to be Visualized
            size (int): Maximum number of data-points/rows to be visualized
        """
        super().__init__()

        import wandb

        self.wandb = wandb
        self.dataset_name = dataset_name
        self.size = size
        self._run = None
        # Table logging utils
        # self._evalset_table = None
        self._evalset_table_rows = []  # reset after log operation
        self._evalset_table_ref = None  # eval artifact refrence for deduping. Don't reset
        self._row_idx = 0
        self._map_table_row_file_name = {}
        self._table_thing_classes = None
        self._table_stuff_classes = None

        # parsed metadata
        self.thing_class_names = []
        self.thing_index_to_class = {}
        self.stuff_class_names = []
        self.stuff_index_to_class = {}

    def process(self, image_mask_overlap_list, video_name):

        if self.size > 0 and len(self._evalset_table_rows) >= self.size // (comm.get_world_size()):
            return

        table_row_colums = []
        for image_mask_overlap in image_mask_overlap_list:
            table_row = self.wandb.Image(image_mask_overlap)
            table_row_colums.append(table_row)
        table_row_colums = [video_name] + [table_row_colums]
        self._evalset_table_rows.append(table_row_colums)
            

    def evaluate(self):
        
        comm.synchronize()
        table_rows = comm.gather(self._evalset_table_rows)
        if comm.is_main_process():
            if self.wandb.run is None:
                raise Exception(
                    "wandb run is not initialized. Either call wandb.init(...) or set WandbWriter()"
                )
            self._run = self.wandb.run
            table_rows = list(itertools.chain(*table_rows))
            _evalset_table = self._build_evalset_table()
            
            for table_row in table_rows:
                # use reference of table if present
                _evalset_table.add_data(*table_row)
                self._map_table_row_file_name[table_row[0]] = self._row_idx
                self._row_idx = self._row_idx + 1

            self._run.log({self.dataset_name: _evalset_table})
        return super().evaluate()

    def reset(self):
        self._build_dataset_metadata()
        # self._evalset_table = None
        self._row_idx = 0
        self._evalset_table_rows = []
        return super().reset()

    def _build_dataset_metadata(self):
        """
        Builds parsed metadata lists and mappings dicts to facilitate logging.
        Builds a list of metadata for each of the validation dataloaders.

        E.g.
        # set the properties separately
        MetadataCatalog.get("val1").property = ...
        MetadataCatalog.get("val2").property = ...

        Builds metadata objects for stuff and thing classes:
        1. self.thing_class_names/self.stuff_class_names -- List[str] of category names
        2. self.thing_index_to_class/self.stuff_index_to_class-- Dict[int, str]

        """
        meta = MetadataCatalog.get(self.dataset_name)

        # Parse thing_classes
        if hasattr(meta, "thing_classes"):
            self.thing_class_names = meta.thing_classes

        wandb_thing_classes = []
        # NOTE: The classs indeces starts from 1 instead of 0.
        for i, name in enumerate(self.thing_class_names, 1):
            self.thing_index_to_class[i] = name
            wandb_thing_classes.append({"id": i, "name": name})

        self._table_thing_classes = self.wandb.Classes(wandb_thing_classes)

        # Parse stuff_classes
        if hasattr(meta, "stuff_classes"):
            self.stuff_class_names = meta.stuff_classes

        wandb_stuff_classes = []
        for i, name in enumerate(self.stuff_class_names):
            self.stuff_index_to_class[i] = name
            wandb_stuff_classes.append({"id": i, "name": name})
        
        self._table_stuff_classes = self.wandb.Classes(wandb_stuff_classes)

    def _parse_prediction(self, pred):
        """
        Parse prediction of one image and return the primitive martices to plot wandb media files.
        Moves prediction from GPU to system memory.

        Args:
            pred (detectron2.structures.instances.Instances): Prediction instance for the image
            loader_i (int): index of the dataloader being used

        returns:
            Dict (): parsed predictions
        """
        parsed_pred = {}
        if pred.get("instances") is not None:
            pred_ins = pred["instances"]
            parsed_pred["boxes"] = (
                pred_ins.pred_boxes.tensor.tolist()[:10] if pred_ins.has("pred_boxes") else None
            )
            parsed_pred["classes"] = (
                pred_ins.pred_classes.tolist()[:10] if pred_ins.has("pred_classes") else None
            )
            parsed_pred["scores"] = (
                pred_ins.scores.tolist()[:10] if pred_ins.has("scores") else None
            )
            parsed_pred["pred_masks"] = (
                pred_ins.pred_masks.cpu().detach().numpy()[:10]
                if pred_ins.has("pred_masks")
                else None
            )  # wandb segmentation panel supports np
            parsed_pred["pred_keypoints"] = (
                pred_ins.pred_keypoints.tolist()[:10] if pred_ins.has("pred_keypoints") else None
            )

        if pred.get("sem_seg") is not None:
            parsed_pred["sem_mask"] = pred["sem_seg"].argmax(0).cpu().detach().numpy()

        if pred.get("panoptic_seg") is not None:
            # NOTE: handling void labels isn't neat.
            panoptic_mask = pred["panoptic_seg"][0].cpu().detach().numpy()
            # handle void labels( -1 )
            panoptic_mask[panoptic_mask < 0] = 0
            parsed_pred["panoptic_mask"] = panoptic_mask

        return parsed_pred

    def _build_evalset_table(self):
        """
        Builds wandb.Table object for logging evaluation

        Args:
            pred_keys (List[str]): Keys in the prediction dict

        returns:
            table (wandb.Table): Table object to log evaluation

        """
        # Current- Use cols. for each detection class score and don't use columns for overlays
        table_cols = ["video_name", "video"]
        table = self.wandb.Table(columns=table_cols)

        return table