INPUT_IMAGE_PATH = "samples/person.jpg"
OUTPUT_IMAGE_PATH = "samples/segm.png"


#################### Code ##############################

from detectron2.config import get_cfg
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer

cfg = get_cfg()
add_densepose_config(cfg)

cfg.merge_from_file("model/densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

predictor = DefaultPredictor(cfg)
img = cv2.imread(INPUT_IMAGE_PATH)
with torch.no_grad():
    outputs = predictor(img)['instances']

results = DensePoseResultExtractor()(outputs)
out_img = Visualizer().visualize(np.zeros(img.shape, dtype=np.uint8), results)
cv2.imwrite(OUTPUT_IMAGE_PATH, out_img)