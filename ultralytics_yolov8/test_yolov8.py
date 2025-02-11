"""
batch_size=16
im_size=800
"""
# with open('flag.txt','r') as f:
#    x=f.read()
#    if x =='':
#        x=0
#    if int(x)>0:
#        exit()
# with open('flag.txt','w') as f:
#    x=int(x)+1
#    f.write(str(x))
import os
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
current_file_path = os.path.abspath(__file__)  
current_file_directory= os.path.dirname(current_file_path)
# current_file_directory = r'C:\Repo\ultralytics_yolov8'
import platform
import sys
sys_name = platform.system()
if sys_name == "Windows":
    pass
    # sys.path.pop(0)
    # sys.path.pop(0)
    # sys.path.insert(0, r'C:\Users\15172\.conda\envs\label\Lib\site-packages\ultralytics')
    # sys.path.insert(0, r'E:\PHD\WeedDetection\ultralytics_old\ultralytics')
else:
    # sys.path.insert(0, '/mnt/e/PHD/WeedDetection/ultralytics_yolov8')
    pass
from ultralytics import YOLO
sys.path.insert(0, current_file_directory)

import config as cfg
cfg.is_val = True

import yaml
from os.path import join
target_name = cfg.target_name

# self station
PC_name = 'station'

for i in range(1, len(sys.argv)):
    print('argument:', i, 'value:', sys.argv[i])
if len(sys.argv) > 1:
    PC_name = sys.argv[1]



if sys_name == "Windows":
    #cfg_file = r"E:\PHD\WeedDetection\ultralytics\ultralytics\models\v8\yolov8l_weed.yaml"
    train_num = 0
    # model_dir_root = r'D:\BoyangDeng\WeedDetection\ultralytics\runs\detect\data2021_subsample_add_I3Net\train2'
    # factor='platform_phone'
    # start_t = 50
    # end_t = 100
    # model_dir_root = rf'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train2017_{factor}_{start_t}_{end_t}'
    # model_dir_root = rf'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train2017_{factor}'
    
    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\train_blueberry_rep1'
    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\denseblueberry\train_blueberry_1560\train_n'
    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\old\dense_blueberry\sample140\train2'
    # model_dir_root = r'.\runs\detect\train_blueberry_1560\train_m.2'
    # model_dir_root = r'.\runs\detect\train_blueberry_3520\train2_rep3'
    # model_dir_root = r'C:\Users\15172\OneDrive_MSU\Project\BlueberryDetectionAPP\Models\yolov8\700images_v8130_l_data2223_rep00'
    # model_dir_root = r'.\runs\detect\denseblueberry\train_blueberry_1280\train_blueberry_rep3_part2_hw_1280_720_yolov8l'

    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\denseblueberry\iOSAPP\train_blueberry_1560\train_m.2'
    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\denseblueberry\iOSAPP\train_blueberry_3520\train2_rep3'
        
    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\denseblueberry\train_blueberry_3520\high_accuracy\train2_rep1'
    
    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\stable_diffusion_controlnet\with_basic_data_aug\yolov8l_24epochs_960_train_train2017_real_and_synthetic_from_detection_17171_0_val_val2017_seed100'
    # model_dir_root = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\weed_stable_diffusion_IPAdapter\yolov8l_24epochs_1024_train_mask_soil_with_weeds_no_overlap_generated_v2_val_val2017'
    # model_dir_root=r"D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\SSOD_2\yolov8l_16epochs_960_train_train2017_real_val_val2017_semisupervised"
    model_dir_root=r"D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\yolov8l_pretrain_ultralytic_24epochs_800_train_bg_weed_images_basic_val_val2017_20250209_210231"
    
    if train_num <= 0:
        model_dir = join(model_dir_root, 'weights')
    else:
        model_dir = join(model_dir_root + str(train_num), 'weights')
    # weed_2022 weed_2022_test  weed_2022_test_video weed_2022_test_all
    # weed_2021_test weed_2021_test_all
    data_dir_root = join(current_file_directory, r'.\ultralytics\yolo\data\datasets')
    if target_name == 'blueberry_dense':
        data_dir = join(data_dir_root, 'dense_blueberry_test.yaml')
    elif target_name == 'cottonboll_dense':
        # dense_blueberry_test dense_cotton_boll_test
        data_dir = join(data_dir_root, 'dense_cotton_boll_test.yaml')
    elif target_name == 'weed2021':
        # weed_2021_test_all weed_2022_test_all
        data_dir = join(data_dir_root, 'weed_2021_test_rep2.yaml')
    elif target_name == 'weed2022':
        data_dir = join(data_dir_root, 'weed_2022_test.yaml')
    elif target_name == 'weed2023':
        data_dir = join(data_dir_root, 'weed_2023_test.yaml')
    elif target_name == 'weed_all':
        data_dir = join(data_dir_root, 'weed_all_test.yaml')
    elif target_name == 'weed_lambsquarters':
        # weed_lambsquarters_test weed_lambsquarters_test_density_0.0_0.25 weed_lambsquarters_test_density_0.25_0.5 weed_lambsquarters_test_density_0.5_0.75 weed_lambsquarters_test_density_0.75_1.0
        #data_dir = join(data_dir_root, 'weed_lambsquarters_test_0_20.yaml')
        #data_dir = join(data_dir_root, 'weed_lambsquarters_test_20_200.yaml')
        data_dir = join(data_dir_root, 'weed_lambsquarters_test.yaml')
    elif target_name == 'COCO':
        data_dir = join(data_dir_root, '7sku_original_im_to_generation_train_test.yaml')
    elif target_name == 'weed_stable_diffusion':
        data_dir = join(data_dir_root, 'weed_stable_diffusion.yaml')
    elif target_name == 'weed_stable_diffusion_controlnet':
        data_dir = join(data_dir_root, 'weed_stable_diffusion_controlnet_test.yaml')
    elif target_name == 'weed_stable_diffusion_IPAdapter':
        data_dir = join(data_dir_root, 'weed_stable_diffusion_IPAdapter_test.yaml')
    elif target_name == 'weed_semi_supervised':
        data_dir = join(data_dir_root, 'weed_semi_supervised_test.yaml')
        
else:
    # yolov8l_weed yolov8s_weed
    root_dir = '/mnt/d/BoyangDeng/Detection/ultralytics_yolov8'
    cfg_file = join(root_dir, "ultralytics/models/v8/yolov8l_weed.yaml")
    train_num = 4
    model_dir = join(root_dir, 'runs/detect/data2021_subsample/replication1/train' + str(train_num) + '/weights')
    # weed_test_dataset_2021_linux weed_test_dataset_2022_linux
    # weed_test_dataset_2021_linux_all weed_test_dataset_2022_linux_all
    data_dir = join(root_dir, 'ultralytics/yolo/data/datasets/weed_test_dataset_2021_linux.yaml')

"""
conda activate pytorch
cd /mnt/e/Repo/Biang/Graphics/WeedDetection
python test_yolov8.py
windows/linux may cannot share dataset xxx.cache
Logging results to /mnt/e/Repo/Biang/runs/detect/train
E:\Repo\Biang\runs\detect
"""
print('data config:', os.path.split(data_dir)[1])
with open(data_dir, 'r', encoding='UTF-8') as file:
    data_info = yaml.safe_load(file)

if PC_name == 'self':

    # train2 train2_rep2 train2_rep3
    #model_dir = r'D:\BoyangDeng\ultralytics_yolov8\runs\detect\old\dense_blueberry\sample140\train2_rep2\weights'
    # model_dir = r'D:\BoyangDeng\Detection\ultralytics_yolov8\runs\detect\weed_all\train2\weights'

    #model_dir = r'D:\BoyangDeng\ultralytics_yolov8\runs\detect\roi\400\train2\weights'

    #model_dir = r'D:\BoyangDeng\ultralytics_yolov8\runs\detect\roi\train2\weights'
    #model_dir = r'D:\BoyangDeng\ultralytics_yolov8\runs\detect\train\weights'
    pass

# model = YOLO(cfg_file)  # build a new model from scratch

print('data config:', os.path.split(data_dir)[1])

# best/last
model_path = os.path.join(model_dir, 'best.pt')
print('model_path:', model_path)

if cfg.test_use_roi_model:
    #roi_model_dir = r'D:\BoyangDeng\ultralytics_yolov8\runs\detect\roi\400\train2_rep2\weights'
    roi_model_dir = r'D:\BoyangDeng\ultralytics_yolov8\runs\detect\denseblueberry\roi\train2\weights'
    roi_model_path = os.path.join(roi_model_dir, 'best.pt')
    cfg.roi_model = YOLO(roi_model_path)

conf=cfg.val_bbox_conf
iou = cfg.nms_iou
verbose = cfg.verbose
model = YOLO(model_path)
model.model.names = data_info['names']
cfg.label_idx_names = data_info['names']
# 16/1
#plots=True
batch = 4
results = model.val(data=data_dir, workers=0, batch=batch, conf=conf, iou=iou, verbose=verbose)
#results = model("https://ultralytics.com/images/bus.jpg")

# Export the model
# model.export(format="onnx")
