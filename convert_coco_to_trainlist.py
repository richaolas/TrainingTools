"""

convert coco to trainlist


"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import random
import fire
import json


'''as 
img_data = {
    'image':path
  #'pixel_data': bytes,
  'bboxes': [[0.2, .3, .4, .5], [.1, .3, .4, .5]],
  'labels': [23, 34]
  'class/text':['xx', 'yyy']
  'class/label': [0,2]
}

'''


def drawbox(img, box):
    (img_h, img_w, img_c) = img.shape
    (x, y, w, h) = box
    p0 = (int((x - w/2) * img_w), int((y - h/2) * img_h))
    p1 = (int((x + w/2) * img_w), int((y + h/2) * img_h))
    cv2.rectangle(img, p0, p1, (255, 255, 255), 1)


def convert_boxes_labels(ct_boxes, ratio):
    boxlab = np.zeros((len(ct_boxes) * 6 + 1), np.float32)
    for i in range(len(ct_boxes)):
        box = ct_boxes[i]
        boxlab[i * 6 + 0] = 0
        boxlab[i * 6 + 1: i * 6 + 5] = box
        boxlab[i * 6 + 5] = i + 1
    boxlab[len(ct_boxes) * 6 + 0] = ratio
    return boxlab


def resize_mask(mask, size):
    (dst_h, dst_w) = size
    mk_resize = np.zeros((dst_h, dst_w, 1), np.uint8)
    (img_h, img_w, img_c) = mask.shape
    for r in range(dst_h):
        for c in range(dst_w):
            (o_r, o_c) = (int(1.0 * img_h / dst_h * r), int(1.0 * img_w / dst_w * c))
            mk_resize[r, c, 0] = mask[o_r, o_c, 0]
    return mk_resize


def run(coco_dir):
    (dst_h, dst_w) = [320, 320]
    #dataTypes = ['train', 'val']
    #years = ['2014', '2017']
    dataTypes = ['train']
    years = ['2017']

    # dataTypes = ['val']
    # years = ['2017']

    dataDir = coco_dir
    for dataType in dataTypes:
        for year in years:
            imageSet = dataType + year
            ann_path = '{}/annotations/instances_{}.json'.format(
                dataDir, imageSet)
            coco = COCO(ann_path)

            target_f = 'coco_{}_{}.txt'.format(dataType, year)
            print('generating {}'.format(target_f))
            target = open(target_f, 'w')

            target_file = open(target_f+'.json', 'w')
            target_file.write('[\n')
            # for idx, item in enumerate(totoal):
            #     if idx > 0:
            #
            #         f.write(',\n' + json.dumps(item))
            #     else:
            #         f.write(json.dumps(item))
            # f.write(']')
            #
            # f.close()

            # display COCO categories and supercategories
            cats = coco.loadCats(coco.getCatIds())
            imgIds = coco.getImgIds()

            random.shuffle(imgIds)
            count_id = 0

            for imgId in imgIds:

                annotation = {}

                count_id = count_id+1
                img = coco.loadImgs(imgId)[0]

                img_path = '%s/%s' % (imageSet, img['file_name'])
                print(img_path)

                annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
                anns = coco.loadAnns(annIds)

                anno_part = ''
                boxs = []
                label_ids = []
                label_texts = []
                for ann in anns:
                    box = ann['bbox']
                    boxs.append(box)
                    box = [str(i) for i in box]
                    label_id = str(ann['category_id'])
                    one_part = ','.join(box + [label_id])
                    anno_part += ' {}'.format(one_part)
                    
                    label_ids.append(ann['category_id'])
                    #ret = coco.loadCats(ann['category_id'])
                    label_texts.append(coco.loadCats(ann['category_id'])[0]['name'])
                    
                annotation['image'] = img_path
                annotation['width'] = img['width']
                annotation['height'] = img['height']
                annotation['boxs'] = boxs
                #annotation['.data'] =
                annotation['class-text'] = label_texts
                annotation['class-id'] = label_ids

                one_line = '{}{}\n'.format(img_path, anno_part)
                target.write(one_line)

                if count_id == 1:
                    target_file.write(json.dumps(annotation))
                else:
                    target_file.write(',\n' + json.dumps(annotation))
    target_file.write(']\n')
    print('done!')


if __name__ == '__main__':
    fire.Fire(run)
