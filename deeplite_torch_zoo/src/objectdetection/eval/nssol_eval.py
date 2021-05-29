# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import time

import cv2
import numpy as np
import torch

from deeplite_torch_zoo.src.objectdetection.datasets.nssol import NSSOLDataset
from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.src.objectdetection.eval.metrics import MAP
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.tools import post_process


class NSSOLEval(Evaluator):
    """docstring for NSSOLEval"""

    def __init__(self, model, data_root, visiual=False, net="yolo3", img_size=448):
        data_path = "deeplite_torch_zoo/results/nssol/{net}".format(net=net)
        super(NSSOLEval, self).__init__(
            model=model, data_path=data_path, img_size=img_size, net=net
        )
        self.dataset = NSSOLDataset(data_root, _set="test")
        self.data_root = data_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):

        self.model.eval()
        self.model.cuda()
        results = []
        start = time.time()
        avg_loss = 0
        for image, labels, _, _ in self.dataset:

            print("Parsing batch: {}/{}".format(img_idx, len(self.dataset)), end="\r")
            bboxes_prd = self.get_bbox(image)
            if len(bboxes_prd) == 0:
                bboxes_prd = np.zeros((0, 6))

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            detections = {"bboxes": [], "labels": []}
            detections["bboxes"] = bboxes_prd[:, :5]
            detections["labels"] = bboxes_prd[:, 5]

            gt = {"bboxes": labels[:, :4], "labels": labels[:, 4]}
            results.append({"detections": detections, "gt": gt})
        print("validation loss = {}".format(avg_loss))
        # put your model in training mode back on

        mAP = MAP(results, self.dataset.num_classes)
        mAP.evaluate()
        ap = mAP.accumlate()
        mAP_all_classes = np.mean(ap)

        for i in range(0, ap.shape[0]):
            print("{:_>25}: {:.3f}".format(self.dataset.classes[i], ap[i]))

        print("(All Classes) AP = {:.3f}".format(np.mean(ap)))

        ap = ap[ap > 1e-6]
        ap = np.mean(ap)
        print("(Selected) AP = {:.3f}".format(ap))

        return ap  # Average Precision  (AP) @[ IoU=050 ]


def yolo_eval_nssol(model, data_root, device="cuda:0", net="yolov3", img_size=448, **kwargs):

    mAP = 0
    result = {}
    model.to(device)
    with torch.no_grad():
        mAP = NSSOLEval(model, data_root, net=net, img_size=img_size).evaluate()
        result["mAP"] = mAP

    return result
