import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/p2-siam/weights/best.pt')
    model.val(data='data/data.yaml',split='test', name='p2-siam', batch=1)