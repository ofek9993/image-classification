import torchvision
import torch
import numpy as np
from torchvision import transforms as T


# Loading our model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Need to put the model in evaluation mode

# All the items our model can recognize
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic-light', 'fire-hydrant', 'N/A', 'stop-sign',
    'parking-meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports-ball',
    'kite', 'baseball-bat', 'baseball-glove', 'skateboard', 'surfboard', 'tennis-racket',
    'bottle', 'N/A', 'wine-glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted-plant', 'bed', 'N/A', 'dining-table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell-phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy-bear', 'hair-drier', 'toothbrush', 'hair-brush'
]

#get model prediction
def get_prediction(img, threshold=0.8):
    try:
        # the photo are in numpy array form and for pytorch to read it it needs to be in tensor form
        transform = T.Compose([T.ToTensor()])
        temp_image = transform(img)
        # then we make a prediction
        pred = model([temp_image])  # We have to pass in a list of images
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Bounding Boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_box = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_box, pred_class, pred_score
    except IndexError:
        return 1
        # If we cannot find a prediction then we return 1

