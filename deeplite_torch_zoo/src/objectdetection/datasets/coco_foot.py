
import os
import time
import numpy as np
import torch
import json
import copy

from PIL import ImageFile
from PIL import Image
from torchvision.datasets import CocoDetection
from collections import defaultdict

from pycocotools.coco import COCO


class COCOFoot(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            _annotations = []
            for ann in dataset["annotations"]:
                left_kp = np.array(ann["keypoints"]).reshape((-1, 3))[-6:-3, :]
                left_bbox = keypoints_to_xyxy(left_kp)[0]
                c_ann_left = copy.deepcopy(ann)
                c_ann_left["bbox"] = [left_bbox[0], left_bbox[1], left_bbox[2] - left_bbox[0], left_bbox[3] - left_bbox[1]]
                c_ann_left["id"] = len(_annotations)
                #c_ann_left["category_id"] = 2
                _annotations.append(c_ann_left)

                right_kp = np.array(ann["keypoints"]).reshape((-1, 3))[-3:, :]
                right_bbox = keypoints_to_xyxy(right_kp)[0]
                c_ann_right = copy.deepcopy(ann)
                c_ann_right["bbox"] = [right_bbox[0], right_bbox[1], right_bbox[2] - right_bbox[0], right_bbox[3] - right_bbox[1]]
                c_ann_right["id"] = len(_annotations)
                #c_ann_right["category_id"] = 2
                _annotations.append(c_ann_right)
            dataset["annotations"] = _annotations
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()


def xywh_to_xyxy(bbox):
    bbox[:, 2] += bbox[:, 0]
    bbox[:, 3] += bbox[:, 1]
    return bbox

def xyxy_to_xywh(bbox):
    bbox[:, 2] -= bbox[:, 0]
    bbox[:, 3] -= bbox[:, 1]
    return bbox


from PIL import Image, ImageDraw, ImageFont
def drawBoundingBoxes(imageData, bboxes, imageOutputPath):
    img = ImageDraw.Draw(imageData)
    for bbox in bboxes:
        img.rectangle(list(bbox), fill=None, outline=None, width=1)
    imageData.save(imageOutputPath)


def keypoints_to_xyxy(kp):
    if min(kp[:, 0]) == 0 or min(kp[:, 1]) == 0:
        return np.zeros((1, 4))
    xyxy = [min(kp[:, 0]), min(kp[:, 1]), max(kp[:, 0]), max(kp[:, 1])]
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    _margin = min(w, h)
    _margin = min(_margin, 10)
    _margin = max(_margin, 5)
    xyxy[0] -= _margin
    xyxy[1] -= _margin
    xyxy[2] += _margin
    xyxy[3] += _margin
    return np.array([xyxy], dtype=np.int)


class CocoFootDetectionBoundingBox(CocoDetection):
    def __init__(
        self,
        img_root,
        ann_file_name,
        num_classes=80,
        transform=None,
        category=1,
        img_size=416,
        classes=[],
        missing_ids=[]
    ):
        super(CocoFootDetectionBoundingBox, self).__init__(img_root, ann_file_name)
        self._tf = transform
        self._img_size = img_size
        self.classes = ["BACKGROUND"] + classes
        self.num_classes = len(self.classes)
        self.missing_ids = missing_ids
        if category == "all":
            self.all_categories = True
            self.category_id = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_id = category
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self._filter_out_non_person_images()


    def _filter_out_non_person_images(self):
        _ids = []
        ids = list(self.coco.imgs.keys())
        for img_id in ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            targets = self.coco.loadAnns(ann_ids)
            person_exists = False
            for target in targets:
                if target["category_id"] == 1:
                    person_exists = True
                    break
            if person_exists:
                _ids.append(img_id)
        self.ids = _ids


    def __getitem__(self, index):
        """
        return:
            label_tensor of shape nx6, where n is number of labels in the image and x1,y1,x2,y2, class_id and confidence.
        """
        img, targets = super(CocoFootDetectionBoundingBox, self).__getitem__(index)
        labels = []
        for target in targets:
            left_kp = np.array(target["keypoints"]).reshape((-1, 3))[-6:-3, :]
            left_bbox = keypoints_to_xyxy(left_kp)

            #bbox = torch.tensor(target["bbox"], dtype=torch.float32)  # in xywh format
            right_kp = np.array(target["keypoints"]).reshape((-1, 3))[-3:, :]
            right_bbox = keypoints_to_xyxy(right_kp)
            bbox = np.concatenate((left_bbox, right_bbox), axis=0)
            #drawBoundingBoxes(img, bbox, f"{index}.jpg")
            bbox = xyxy_to_xywh(bbox)
            bbox = torch.tensor(bbox, dtype=torch.float32)


            category_id = target["category_id"]
            conf = torch.tensor([[1.0, 1.0]])
            category_id = torch.tensor([[1, 1]])
            label = torch.cat((bbox, torch.transpose(category_id, 0, 1), torch.transpose(conf, 0, 1)), dim=1)
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, 6))
        del labels

        if self._tf == None:
            return np.array(img), None, None, self.ids[index]
        transformed_img_tensor, label_tensor = self._tf(self._img_size)(
            img, label_tensor
        )
        label_tensor = xywh_to_xyxy(label_tensor)
        return (
            transformed_img_tensor,
            label_tensor,
            label_tensor.size(0),
            self.ids[index],
        )


    def collate_img_label_fn(self, sample):
        images = []
        labels = []
        lengths = []
        labels_with_tail = []
        img_ids = []

        max_num_obj = 0
        for image, label, length, img_id in sample:
            images.append(image)
            labels.append(label)
            lengths.append(length)
            max_num_obj = max(max_num_obj, length)
            img_ids.append(torch.tensor([img_id]))
        for label in labels:
            num_obj = label.size(0)
            zero_tail = torch.zeros(
                (max_num_obj - num_obj, label.size(1)),
                dtype=label.dtype,
                device=label.device,
            )
            label_with_tail = torch.cat((label, zero_tail), dim=0)
            labels_with_tail.append(label_with_tail)
        image_tensor = torch.stack(images)
        label_tensor = torch.stack(labels_with_tail)
        length_tensor = torch.tensor(lengths)
        img_ids_tensor = torch.stack(img_ids)
        return image_tensor, label_tensor, length_tensor, img_ids_tensor


    def _delete_coco_empty_category(self, old_id):
        """The COCO dataset has 91 categories but 11 of them are empty.
        This function will convert the 80 existing classes into range [0-79].
        Note the COCO original class index starts from 1.
        The converted index starts from 0.
        Args:
            old_id (int): The category ID from COCO dataset.
        Return:
            new_id (int): The new ID after empty categories are removed."""
        starting_idx = 1
        new_id = old_id - starting_idx
        for missing_id in self.missing_ids:
            if old_id > missing_id:
                new_id -= 1
            elif old_id == missing_id:
                raise KeyError(
                    "illegal category ID in coco dataset! ID # is {}".format(old_id)
                )
            else:
                break
        return new_id


    def add_coco_empty_category(self, old_id):
        """The reverse of delete_coco_empty_category."""
        starting_idx = 1
        new_id = old_id + starting_idx
        for missing_id in self.missing_ids:
            if new_id >= missing_id:
                new_id += 1
            else:
                break
        return new_id