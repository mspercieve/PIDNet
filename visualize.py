import cv2 
import torch    
import numpy as np     
import os    
import argparse
import os
import torch.optim
from torch.nn import functional as F
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate

#device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0')
print(device)

colormap = torch.zeros(256,3).to(device)
colormap[0] = torch.tensor([128, 64, 128])
colormap[1] = torch.tensor([244, 35, 232])
colormap[2] = torch.tensor([70, 70, 70])
colormap[3] = torch.tensor([102, 102, 156])
colormap[4] = torch.tensor([190, 153, 153])
colormap[5] = torch.tensor([153, 153, 153])
colormap[6] = torch.tensor([250, 170, 30])
colormap[7] = torch.tensor([220, 220, 0])
colormap[8] = torch.tensor([107, 142, 35])
colormap[9] = torch.tensor([152, 251, 152])
colormap[10] = torch.tensor([70, 130, 180])
colormap[11] = torch.tensor([220, 20, 60])
colormap[12] = torch.tensor([255, 0, 0])
colormap[13] = torch.tensor([0, 0, 142])
colormap[14] = torch.tensor([0, 0, 70])
colormap[15] = torch.tensor([0, 60, 100])
colormap[16] = torch.tensor([0, 80, 100])
colormap[17] = torch.tensor([0, 0, 230])
colormap[18] = torch.tensor([119, 11, 32])

bd_colormap = torch.zeros(9,3).to(device)
bd_colormap[1] = torch.tensor([255,255,255]) #흰색   0~0.2
bd_colormap[2] = torch.tensor([192,192,192]) #회색 0.2~0.5
bd_colormap[3] = torch.tensor([102,255,178]) #연두색 0.5~0.8
bd_colormap[4] = torch.tensor([255,153,51]) #파랑색 (Boundary로 예측중 GT가 아닌 것)
bd_colormap[5] = torch.tensor([0,0,255]) #빨간색 GT!=BD pred(완전히 포착을 못함)
bd_colormap[6] = torch.tensor([0,128,255]) #주황색 (심하게 덜 예측)
bd_colormap[7] = torch.tensor([255,0,255]) #핑크색 (덜 예측)
bd_colormap[8] = torch.tensor([0,0,0]) # 검은색 GT= BD pred

label_pred_colormap = torch.zeros(3,1024,2048)
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args





batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=config.TRAIN.SHUFFLE,
    num_workers=config.WORKERS,
    pin_memory=False,
    drop_last=True)


if config.LOSS.USE_OHEM:
    sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    thres=config.LOSS.OHEMTHRES,
                                    min_kept=config.LOSS.OHEMKEEP,
                                    weight=train_dataset.class_weights)

else:
    sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                weight=train_dataset.class_weights)

bd_criterion = BondaryLoss()





test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.TEST_SET,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=False,
                    flip=False,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TEST.BASE_SIZE,
                    crop_size=test_size)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size= 3,
    shuffle=False,
    num_workers=config.WORKERS,
    pin_memory=False)

imgnet = 'imagenet' in config.MODEL.PRETRAINED
model = models.pidnet_bcf2.PIDNet_bcf2(m=2, n=3, num_classes= config.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True).to(device)


model_state_file = os.path.join('/home/mvpserverfive/minseok/PIDNet/output/cityscapes/pidnet_small_cityscapes/', 'final_state.pt')
#model_state_file = os.path.join('/home/mvpserverfive/minseok/PIDNet/output/cityscapes/pidnet_small_cityscapes_trainval/', 'best.pt')
if os.path.isfile(model_state_file):
    print(1)
    checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
    #print(checkpoint['epoch'])
    #dct = checkpoint['state_dict']
    model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint.items() if k.startswith('model.')})

for i_iter, batch in enumerate(testloader, 0):
    image, label, bd_gts, _, _ = batch
    size = label.size()
    image = image.cuda()
    print('img_size',image.size(), 'label_size',label.size(), 'bd_gt_size',bd_gts.size())
    print('2')
    label = label.long().cuda()
    print('3')
    bd_gts = bd_gts.float().cuda()
    print('4')
    seghead_pred, output_pred, boundary_pred = model(image)
    print(seghead_pred.size(), output_pred.size(), boundary_pred.size())
    
    output_pred = F.interpolate(output_pred, size=[1024,2048], mode='bilinear', align_corners=False)
    output_pred = F.softmax(output_pred, dim=1)
    output_pred = output_pred.argmax(dim=1)

    pred_colormap = torch.zeros((3,1024,2048)).to(device)
    label_colormap = torch.zeros_like(pred_colormap).to(device)
    boundary_map = torch.zeros((3,1024,2048)).to(device)
    bdmap = torch.zeros((1,1024,2048)).to(device)
    '''
    for x_index, x_value in enumerate(output_pred[0],0):
        for y_index, y_value in enumerate(x_value, 0):
            pred_colormap[:x_index,y_index]= colormap[y_value]
    print(pred_colormap)
    print('output_pred_size', output_pred.size())
    '''
    for k in range (19):
        pred_colormap[0,output_pred[0]==k]=colormap[k][0]
        pred_colormap[1,output_pred[0]==k]=colormap[k][1]
        pred_colormap[2,output_pred[0]==k]=colormap[k][2]

        label_colormap[0,label[0]==k]=colormap[k][0]
        label_colormap[1,label[0]==k]=colormap[k][1]
        label_colormap[2,label[0]==k]=colormap[k][2]

    label_colormap_np = label_colormap.permute(1,2,0)
    label_colormap_np = label_colormap_np.cpu().detach().numpy()
    label_colormap_filename = "/home/mvpserverfive/minseok/PIDNet/visualize/label/label_color_%d.png" %i_iter
    
    label_pred_colormap = label_colormap
    label_pred_colormap[0,label[0]!=output_pred[0]]= 255
    label_pred_colormap[1,label[0]!=output_pred[0]] = 255
    label_pred_colormap[2, label[0]!=output_pred[0]] = 255
    label_pred_colormap[0,label[0]==255]=0
    label_pred_colormap[1,label[0]==255]=0
    label_pred_colormap[2,label[0]==255]=0
    
    #print(label_pred_colormap)
    label_pred_colormap = label_pred_colormap.permute(1,2,0)
    label_pred_colormap =label_pred_colormap.cpu().detach().numpy()
    label_pred_colormap_filename = "/home/mvpserverfive/minseok/PIDNet/visualize/label_pred/label_pred_color_%d.png" %i_iter
    cv2.imwrite(label_pred_colormap_filename, label_pred_colormap)
    

    image = image[0].permute(1,2,0)
    image = image.cpu().detach().numpy()
    image_filename = "/home/mvpserverfive/minseok/PIDNet/visualize/img/img_%d.png" %i_iter

    pred_colormap = pred_colormap.permute(1,2,0)    
    pred_colormap = pred_colormap.cpu().detach().numpy()
    pred_colormap_filename = "/home/mvpserverfive/minseok/PIDNet/visualize/pred/pred_color_%d.png" %i_iter

    cv2.imwrite(label_colormap_filename, label_colormap_np)
    cv2.imwrite(image_filename, image)
    cv2.imwrite(pred_colormap_filename, pred_colormap)
    
    
    boundary_upsize = F.interpolate(boundary_pred, size=[1024,2048], mode='bilinear', align_corners= False)
    boundary_output = F.sigmoid(boundary_upsize)
    print('x',boundary_output.size())
    boundary_output=boundary_output[0]
    print('y',bd_gts.size())
    bd_gts=bd_gts[0].unsqueeze(dim=0)
    print('x',boundary_output.size())
    print('y',bd_gts.size())
    bdmap[boundary_output<0.2]=1 #흰색
    bdmap[boundary_output>0.2]=2 #주황색
    bdmap[boundary_output>0.5]=3 #초록색
    bdmap[boundary_output>0.8]=4 #파란색 (0.8이상이지만 GT가 아닌것)
    bdmap[(bdmap*bd_gts)==1]=5 #빨간색 (GT중 0.2 이하로 예측)
    bdmap[(bdmap*bd_gts)==2]=6 #검은색 (GT중 0.2~0.5로 예측)
    bdmap[(bdmap*bd_gts)==3]=7 #보라색 (GT중 0.5~0.8로 예측)
    bdmap[(bdmap*bd_gts)==4]=8 #검은색 (잘 예측)
    
    
    # 0.2이하 = 흰색 / 0.2~0.5= 주황색/ 0.5~0.8=초록색 / 0.8~1 = 하늘색 / GT=검은색
    for k in range(1,9):
        boundary_map[0,bdmap[0]==k]=bd_colormap[k][0]
        boundary_map[1,bdmap[0]==k]=bd_colormap[k][1]
        boundary_map[2,bdmap[0]==k]=bd_colormap[k][2]

    boundary_map = boundary_map.permute(1,2,0)
    boundary_map = boundary_map.cpu().detach().numpy()
    boundary_map_filename = "/home/mvpserverfive/minseok/PIDNet/visualize/boundary_map/boundary_map_%d.png" %i_iter
    cv2.imwrite(boundary_map_filename, boundary_map)
    
    

    

