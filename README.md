# Video-kMaX (WACV 2024)

This is an official implementation of our WACV 2024 paper: [Video-kMaX](https://arxiv.org/pdf/2304.04694.pdf)

## Installation
The code-base is verified with pytorch==1.12.1, torchvision==0.13.1, cudatoolkit==11.3, and detectron2==0.6,
please install other libiaries through *pip3 install -r requirements.txt*

Please refer to [Mask2Former's script](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md) for data preparation.


## Model Zoo


### VIPSeg VPS

### KITTI-STEP VPS

## Citing Video-kMaX

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

*   Video-kMaX:

(current BibTeX is for arxiv. We will replace it with WACV version after proceeding)

```
@misc{shin2023videokmax,
      title={Video-kMaX: A Simple Unified Approach for Online and Near-Online Video Panoptic Segmentation}, 
      author={Inkyu Shin and Dahun Kim and Qihang Yu and Jun Xie and Hong-Seok Kim and Bradley Green and In So Kweon and Kuk-Jin Yoon and Liang-Chieh Chen},
      year={2023},
      eprint={2304.04694},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```