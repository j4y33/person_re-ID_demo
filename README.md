# Tracking using person re-identification

Learning Generalisable Omni-Scale Representations for Person Re-Identification
https://arxiv.org/abs/1910.06827

## Requirements

1. pytorch 1.3
2. numpy 1.17, Pillow 6.2, opencv-python 4.1
3. [mmdetection](https://github.com/open-mmlab/mmdetection) 1.0rc1+cd0d37c
4. [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) 1.0.9
5. dvc 0.77

```
mkdir checkpoints
wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth
https://mega.nz/#!YTZFnSJY!wlbo_5oa2TpDAGyWCTKTX1hh4d6DvJhh_RUA2z6i_so
```

All models and videos for testing are here:
`s3://dsirf-prototypes/person-reid-demo`
## Installation

```bash
pip install -r requirements.txt
pip install requirements-model.txt
dvc pull
```

## Useful resources

https://github.com/NEU-Gou/awesome-reid-dataset
https://github.com/bismex/Awesome-person-re-identification
