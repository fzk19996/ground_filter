import sys
import os
import numpy as np
sys.path.append('/home/kaikai/code/point_cloud/pc_superposition/utils')
from iou import calc_iou

predict_path = '/home/kaikai/code/RandLA-Net-master/test/sequences'
kitti_path = '/home/kaikai/dataset/vo/sequences'
sequence = '04'
iou_list = []
label_file = '000000.label'
label_path = os.path.join(predict_path, sequence, 'predictions')
gt_path = os.path.join(kitti_path, sequence.zfill(2), 'labels', label_file)
lable_data = np.fromfile(os.path.join(label_path, label_file), dtype=np.uint32)
pred = np.where(lable_data==40, 40, 0)
iou = calc_iou(lable_data, gt_path)
iou_list.append(iou)
for label_file in os.listdir(label_path):
    gt_path = os.path.join(kitti_path, sequence.zfill(2), 'labels', label_file)
    lable_data = np.fromfile(os.path.join(label_path, label_file), dtype=np.uint32)
    pred = np.where(lable_data==40, 40, 0)
    iou = calc_iou(lable_data, gt_path)
    iou_list.append(iou)
print('end')