import os
import cv2 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model.loss import CrossEntropyLabelSmooth
from model import convnext
from dataset_muti import MyDataset
from urllib.request import urlopen
import torch.nn as nn

num_classes1 = 2
num_classes2 = 4
num_classes3 = 14
target_name = {
'name1':['TFT','CF'],
'name2': ['NP','UP','OP','INT'],
'name3': ['CF REPAIR FAIL','PI SPOT-WITH PAR','POLYMER','GLASS BROKEN','PV-HOLE-T','CF DEFECT','CF PS DEFORMATION','FIBER','AS-RESIDUE-E','LIGHT METAL','GLASS CULLET','ITO-RESIDUE-T','M1-ABNORMAL','ESD']
}

'''mdoelsave.pth location'''
modelName = './pth/FMA_Tiny/convnet_final_B.pth'
csv_save_filename="B_convnet_final.csv"

detect_dir = './data/muti/detect/'
result_dir = './result/'
imgpath = './data/muti/detect/' #detect圖片所在位置
CAM_RESULT_PATH = './result/cam_pic/detect/'  #單熱力圖儲存路徑
CAM_FALSE_PATH = './result/cam_pic/detect/detect_wrong/'  #分類錯誤的熱力圖與原圖儲存路徑
CAM_RIGHT_PATH = './result/cam_pic/detect/detect_right/'  #分類正確的熱力圖與原圖儲存路徑


device = torch.device("cuda:0" if torch.cuda.is_available() and 'gpu' == 'gpu' else 'cpu')

def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return
makedirs(CAM_RESULT_PATH)
makedirs(CAM_FALSE_PATH)
makedirs(CAM_RIGHT_PATH)

# 獲得熱力圖的函數
def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,768,7,7
    output_cam = []
    for idx in class_idx: 
        feature_conv = feature_conv.reshape((nc, h*w))  # [768,7*7]
        cam = torch.mm(weight_softmax[idx].unsqueeze(0),feature_conv)  #(1, 768) * (768, 7*7) -> (1, 7*7)
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = cam_img.detach().cpu().numpy()
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)

        output_cam.append(cam_img)
    return output_cam

''' Mode read'''
## - for single 
#model = convnext.Convnext_single(num_class=num_classes1).to(device)

## for a b 
model = convnext.Convnext_muti(num_class1=num_classes1, num_class2=num_classes2, num_class3=num_classes3).to(device)
##　for c
#model = convnext.Convnext_sharefeature(num_class1=num_classes1, num_class2=num_classes2, num_class3=num_classes3).to(device)

#model = model.to(device)
model.load_state_dict(torch.load(modelName),False)
model_feature = nn.Sequential(*(list(model.stages.children())[:-2])) #熱力圖 ：模型取到外層(3): ConvNeXtStage結束 
model.eval()
model_feature.eval()

#熱力圖：獲得fc層的權重
fc_weights = model.classifier3.weight   #(14,768)

'''data - loader'''
coarse_labels,fine_labels,third_labels = target_name['name1'],target_name['name2'],target_name['name3']
detect_dataset = MyDataset(detect_dir, 'pred',target_name)
detect_generator = DataLoader(dataset=detect_dataset, batch_size=1, shuffle=False)

datadic={}
datalen = len(os.listdir(detect_dir))
dfsave = pd.DataFrame()

for j, (img, img_path) in enumerate(tqdm(detect_generator)):
        if j < datalen :
                print("-----------Number-----:"+str(j))
                #----for test.csv testing----
                batch_x ,imgpath = img.to(device),img_path
                print(imgpath)

        #                print(imgpath)
                ''' Tensor balue'''
                superclass_pred,subclass_pred ,subtwoclass_pred= model(batch_x) 
                #擷取最後一張特徵圖
                feature_map = model_feature(batch_x) 
                #predicted_super = torch.argmax(superclass_pred, dim=1)#tensor([1])
                #predicted_sub = torch.argmax(subclass_pred, dim=1)#tensor([9])
                #predicted_sub tow= torch.argmax(subtwoclass_pred, dim=1)#tensor([9])

                ''' confidence  & classes'''
                ''' - superclasses'''
                probs_super = torch.nn.functional.softmax(superclass_pred, dim=1) 
                super_value,super_index=torch.topk(probs_super,k=2,largest=True) #torch.topk(取出前幾大) , 2取出幾個        
                conf,classes = torch.max(probs_super,1) 
                imgclass= coarse_labels[(classes.item())]
                print('superclass',conf,imgclass)

                ''' - subclasses'''
                probs_sub = torch.nn.functional.softmax(subclass_pred, dim=1)
                sub_value,sub_index=torch.topk(probs_sub,k=4,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                conf_sub,classes_sub = torch.max(probs_sub,1)
                imgclass_sub= fine_labels[(classes_sub.item())]
                print('subclass',conf_sub,imgclass_sub)

                ''' - subtwoclasses'''
                probs_subtwo = torch.nn.functional.softmax(subtwoclass_pred, dim=1)
                subtwo_value,subtwo_index=torch.topk(probs_subtwo,k=5,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                conftwo_sub,classestwo_sub = torch.max(probs_subtwo,1)
                imgclasstwo_sub= third_labels[(classestwo_sub.item())]
                print('subclass',conftwo_sub,imgclasstwo_sub)
                  
                imagename = imgpath[0].split('/')[4]

                '''- 熱力圖'''
                h_x = torch.nn.functional.softmax(subtwoclass_pred, dim=1).data.squeeze()  #tensor([0.9981, 0.0019])
                probs, idx = h_x.sort(0, True)   #按概率從大到小排列(信心度,索引值)
                CAMs = returnCAM(feature_map, fc_weights, idx)  #得到所有預測類別的熱力圖

                img = cv2.imread(img_path[0])
                height, width, _ = img.shape

                ture_index= [(index) for index in range(14) if third_labels[index] == imagename.split('@')[0]][0]
                idx_index= [(index) for index in range(14) if idx[index] == ture_index][0]
        

                heatmap_pred = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM上顯示信心度最大值的class
                heatmap_true = cv2.applyColorMap(cv2.resize(CAMs[idx_index], (width, height)), cv2.COLORMAP_JET) #CAM上顯示True class
                
                ##- 加入文字
                text_pred = '%s %s %.2f%%' % ('Pred:', imgclasstwo_sub, probs[0]*100) 	#信心度最大值的class
                text_true = '%s %s %.2f%%' % ('True:', imagename.split('@')[0], probs[idx_index]*100)  #True Class
 
                result_pred = heatmap_pred * 0.5 + img * 0.5    #比例可以自己調節
                result_true = heatmap_true * 0.5 + img * 0.5    #比例可以自己調節
                
                cv2.putText(result_pred, text_pred, (20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(result_true, text_true, (20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
                
                #cv2.imwrite(CAM_RESULT_PATH +'_camPred.jpg', result_pred)
                #cv2.imwrite(CAM_RESULT_PATH + '_camTrue.jpg', result_true)
                
                
                #----------------熱力圖與原圖-----------------
                #創建畫布(原圖與熱力圖)
                imgfilename =imagename.split('@')[3]
                if imgclasstwo_sub!= imagename.split('@')[0]:
                        img = cv2.resize(img,(width,height))
                        result_pred = cv2.resize(result_pred,(width,height))
                        result_true = cv2.resize(result_true,(width,height))
                        totalimg = np.hstack((img,result_pred))
                        totalimg = np.hstack((totalimg,result_true))
                        cv2.imwrite(CAM_FALSE_PATH + imagename,totalimg)
                else:
                        img = cv2.resize(img,(width,height))
                        result_pred = cv2.resize(result_pred,(width,height))
                        totalimg = np.hstack((img,result_pred))
                        cv2.imwrite(CAM_RIGHT_PATH + imagename,totalimg)

                ''' Get into datadic '''
                output_dic = {
                        #'super_conf':[str(index)[:6] for index in super_value[0].tolist()],
                        'super_class':[coarse_labels[index] for index in super_index[0].tolist()],
                        #'sub_conf':[str(index)[:6] for index in sub_value[0].tolist()],
                        'sub_class':[fine_labels[index] for index in sub_index[0].tolist()],
                        #'subtwo_conf':[str(index)[:6] for index in subtwo_value[0].tolist()],
                        'subtwo_class':[third_labels[index] for index in subtwo_index[0].tolist()],
                        'Layer_1_ans':imgclass,
                        #'Layer_1_conf':str(conf[0].tolist())[:6],
                        'Layer_2_ans':imgclass_sub,
                        #'Layer_2_conf':str(conftwo_sub[0].tolist())[:6],
                        'Layer_3_ans':imgclasstwo_sub,
                        #'Layer_3_conf':str(conf_sub[0].tolist())[:6],
                        'Layer_1_True':imagename.split('@')[2],
                        'Layer_2_True':imagename.split('@')[1],
                        'Layer_3_True':imagename.split('@')[0]
                }
                ''' dataframe concat'''
                datadic[imagename] = output_dic
                df = pd.DataFrame(datadic)
                df = df.T

                if  len(dfsave) == 0 :
                        dfsave = df 
                else :
                        dfsave = pd.concat([df,dfsave],axis=0)

print("-----------Number-----:"+str(j))
'''datasave cleaner'''
index_duplicates = dfsave.index.duplicated()
dfsave = dfsave.loc[~index_duplicates]
#dfsave.reset_index(drop=True,inplace=True)

makedirs(result_dir)
dfsave.to_csv(result_dir+csv_save_filename,index=True,index_label='ImagePath')
print('data_save:'+result_dir+csv_save_filename)
