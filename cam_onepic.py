#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import convnext
import torch.nn as nn
import matplotlib.pyplot as plt

num_class1 = 2 
num_class2 = 4  
num_class3 = 14 

modelName = './pth/FMA_Tiny/convnet_final_B.pth' #更換為訓練好的模型

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

model_ft = convnext.Convnext_muti(num_class1=num_class1, num_class2=num_class2, num_class3=num_class3).to(device)
model_ft.load_state_dict(torch.load(modelName),False)
#讀取最後一層的輸出特徵圖
model_feature = nn.Sequential(*(list(model_ft.stages.children())[:-2])) #模型取到外層(3): ConvNeXtStage結束

class_ = {0:'CF REPAIR FAIL',1:'PI SPOT-WITH PAR',2:'POLYMER',3:'GLASS BROKEN',4:'PV-HOLE-T',5:'CF DEFECT',6:'CF PS DEFORMATION',7:'FIBER',8:'AS-RESIDUE-E',9:'LIGHT METAL',10:'GLASS CULLET',11:'ITO-RESIDUE-T',12:'M1-ABNORMAL',13:'ESD'}
model_ft.eval()
model_feature.eval()

# Display all model layer weights
"""
for name, para in model_ft.named_parameters():
    print('{}: {}'.format(name, para.shape))
"""
#獲得fc層的權重
fc_weights = model_ft.classifier3.weight   #(14,768)

data_transform = {
        "train": transforms.Compose([transforms.RandomAffine(40, scale=(.85, 1.15), shear=0),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomPerspective(distortion_scale=0.2),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    transforms.ToTensor()]),
        "val": transforms.Compose([
                        transforms.Resize((224, 224), Image.LANCZOS),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])}

#---------------單張測試------------------

img_path = './data/muti/detect/CF DEFECT@NP@CF@CF_B4VW1YC-1-3.jpg'  #圖片路徑
CAM_RESULT_PATH = './data/'  #熱力圖的儲存位置

if not os.path.exists(CAM_RESULT_PATH):
    os.mkdir(CAM_RESULT_PATH)

#加載圖像
_, img_name = os.path.split(img_path)
img = Image.open(img_path).convert('RGB')
img_tensor = (data_transform['val'](img)).unsqueeze(0).to(device) #[1,3,224,224]

# 提取最后一層特徵圖
feature_map = model_feature(img_tensor)
print(feature_map.shape) #(1,768)
# 獲得網路輸出
output1, output2, output3= model_ft(img_tensor)
h_x = torch.nn.functional.softmax(output3, dim=1).data.squeeze()  #tensor([0.9981, 0.0019])
probs, idx = h_x.sort(0, True)  #按概率從大到小排列(信心度,索引值)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()
# 獲得所有類別預測結果（含信心值,由大到小）
for i in range(num_class3):
    print('{:.3f} -> {}'.format(probs[i], class_[idx[i]]))  
# 獲得熱力圖的函數
def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,768,7,7
    output_cam = []
    for idx in class_idx: 
        feature_conv = feature_conv.reshape((nc, h*w))  # [768,7*7]
        #weight_softmax[1,768]
        cam = torch.mm(weight_softmax[idx].unsqueeze(0),feature_conv)  #(1, 768) * (768, 7*7) -> (1, 7*7) 
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = cam_img.detach().cpu().numpy()
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)
 
        output_cam.append(cam_img)
    return output_cam
 
# CAMs = returnCAM(features, fc_weights, [idx[0]])  #得到預測信心度最高的類別對應的熱力圖
CAMs = returnCAM(feature_map, fc_weights, idx)  #得到所有預測類別的熱力圖
print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])

# 讀取圖片
img = cv2.imread(img_path)
img = cv2.resize(img,(640,480))
height, width, _ = img.shape

heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM上顯示信心度最大值的class
result = heatmap * 0.6 + img * 0.4    #比例可以自己調節

text = '%s %.2f%%' % (class_[idx[0]], probs[0]*100)  #熱力圖上的文字顯示
cv2.putText(result, text, (20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            color=(123, 222, 238), thickness=1, lineType=cv2.LINE_AA)

image_name_ = 'output'
cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + class_[idx[0]] + '.jpg', result)
print(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + class_[idx[0]] + '.jpg')