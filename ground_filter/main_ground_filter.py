import os
import numpy as np
import KITTIReader as reader
import preProcessor
import glob
import yaml
import open3d
import sys
sys.path.append('/home/kaikai/code/point_cloud/pc_superposition/utils')
from iou import calc_iou

for sequence in (['05','06','07','08','09','10']):
    kitti_path = '/home/kaikai/dataset/vo/sequences'
    infile_path = os.path.join(kitti_path, sequence, "velodyne", "*.bin")
    file_list = sorted(glob.glob(infile_path))
    preProcess = preProcessor.preProcessor()
    iou_list = []
    for file_name in file_list:
        raw_data = reader.load_velo_scan(file_name)
        #print(raw_data.shape[0])
        label_path = os.path.join(kitti_path, sequence, "labels", (file_name.split('/')[-1]).split('.')[0]+'.label')
        _, pc = preProcess.ground_filter_heightDiff(raw_data, 400, 400, 0.3, 0.5)
        CFG = yaml.safe_load(open('config/semantic-kitti.yaml', 'r'))
        colors = np.zeros((pc.shape[0],3))
        d = pc[:, 3]
        tmp = np.where(d<=0)
        pred = np.where(d>0, 0, 40)
        iou = calc_iou(pred, label_path)
        iou_list.append(iou)
    iou_list = np.array(iou_list)
    np.save('result/'+sequence+'.npy', iou_list)
# for i in range(colors.shape[0]):
#     if d[i] == 20:
#         colors[i] = [0, 0, 255]
#     else:
#         colors[i] = [255,0,0]
    
    # colors[i] = CFG['color_map'][label[i]]
# point_cloud = open3d.PointCloud()
# point_cloud.colors = open3d.Vector3dVector(colors)
# point_cloud.points = open3d.Vector3dVector(pc[:,0:3])
# open3d.draw_geometries([point_cloud])
