# YOLOv5 for iOS app hockey puck's detection training 

This forked repo is customized for my iOS app's model training. We use the object detection model to capture the movement of the hockey puck in the live camera.

Mimic `detect.py` details for `.mlpackage --nms`

- `hockeypuck.yaml`
- `test_image.py`
- `test_video.py`

## Train

**on m-chip macOS**

```bash
conda activate hockeypuck
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"

python train.py --img 640 --batch 16 --epochs 50 --data ./project/yaml/hockey.yaml --weights yolov5s.pt --cache --device mps
```

## Export

Export the model as `.mlpackage` that fits the CoreML to Vision package.

```bash
python export.py --weights runs/train/exp/weights/best.pt --include "coreml" --nms
```

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>
