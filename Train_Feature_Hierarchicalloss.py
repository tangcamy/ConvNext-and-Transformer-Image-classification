import sys
import os 
sys.path.append(os.getcwd())
import timm
import torch
import torch.nn as nn

from dataset_muti import MyDataset
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from model.loss import CrossEntropyLabelSmooth
from model import convnext 
from model import transformer
from model.hierarchical_loss import HierarchicalLossNetwork
from model.level_dict import hierarchy,hierarchy_two
import math
import csv
#import os
from tqdm import tqdm
import pandas as pd
modelsavename = 'best_B_240822_16Defect.pth'
modelfinalname = 'final_B_240822_16Defect.pth'
model_select ='convnext'
train_dir = 'data/muti/train'
test_dir = 'data/muti/test'
num_classes1 = 2
num_classes2 = 3
num_classes3 = 16
epoch_num = 100
batch_size = 32
t=10 #warmup
n_t=0.5
lr_rate = 0.001
'''v2-3-12'''
#target_name = {
#'name1':['TFT','CF'],
#'name2': ['NP','UP','OP'],
#'name3': ['AS-RESIDUE-E','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','FIBER','GLASS CULLET','ITO-RESIDUE-T','LIGHT METAL','M1-ABNORMAL','PI SPOT-WITH PAR','POLYMER','PV-HOLE-T']
#}

'''2-3-17'''#
#target_name = {
#'name1':['TFT','CF'],
#'name2': ['NP','UP','OP'],
#'name3': ['ALCV FAIL','AS-RESIDUE-E','AS-RESIDUE-T','CELL REPAIR FAIL','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','FIBER','GLASS CULLET','ITO-ABNORMAL','LIGHT METAL','M2-RESIDUE-P','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER','SPI-POLYMER','V-POLYMER']
#}


'''2-3-16'''#
target_name = {
'name1':['TFT','CF'],
'name2': ['NP','UP','OP'],
'name3': ['ALCV FAIL','AS-RESIDUE-E','CELL REPAIR FAIL','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','FIBER','GLASS CULLET','ITO-ABNORMAL','LIGHT METAL','M2-RESIDUE-P','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER','SPI-POLYMER','V-POLYMER']
}



'''2-3-14'''#
#target_name = {
#name1':['TFT','CF'],
#'name2': ['NP','UP','OP'],
#'name3': ['ALCV FAIL','AS-RESIDUE-E','CELL REPAIR FAIL','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','FIBER','GLASS CULLET','ITO-ABNORMAL','LIGHT METAL','M2-RESIDUE-P','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER']
#}


'''2-3-22'''#
#target_name = {
#'name1':['TFT','CF'],
#'name2': ['NP','UP','OP'],
#'name3': ['ALCV FAIL','ALSR FAIL','AS-RESIDUE-E','AS-RESIDUE-T','CELL REPAIR FAIL','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','FIBER','GLASS CULLET','ITO-ABNORMAL','LIGHT METAL','M1-ABNORMAL','M1-PARTICLE','M2-ABNORMAL','M2-PARTICLE','M2-RESIDUE-P','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER','SPI-POLYMER','V-POLYMER']
#}



def train():
    acc_best = 0
    train_Loss,test_Loss,acc_list=[],[],[]
    train_dataset = MyDataset(train_dir, 'train',target_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(test_dir, 'test',target_name)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    target1_name, target2_name, target3_name = train_dataset.get_target_name()
    Tar1 = [target1_name.index(i) for i in target1_name]
    Tar2 = [target2_name.index(i) for i in target2_name]
    Tar3 = [target3_name.index(i) for i in target3_name]
    print(target1_name, target2_name, target3_name)
    print(Tar1,Tar2,Tar3)
    
    print('****************')

    device = torch.device('cuda')
    if model_select =='convnext':
        model = convnext.Convnext_muti(num_class1=num_classes1, num_class2=num_classes2, num_class3=num_classes3).to(device)
    else:
        model = transformer.Transformer_muti(num_class1=num_classes1, num_class2=num_classes2, num_class3=num_classes3).to(device)
    
    parameters_1 = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters_1, lr=lr_rate, weight_decay=1e-5)
    lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  0.1  if n_t * (1+math.cos(math.pi*(epoch - t)/(epoch_num-t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(epoch_num-t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion1 = CrossEntropyLabelSmooth(num_classes=num_classes1).to(device)
    criterion2 = CrossEntropyLabelSmooth(num_classes=num_classes2).to(device)
    criterion3 = CrossEntropyLabelSmooth(num_classes=num_classes3).to(device)
 
    HLN = HierarchicalLossNetwork(metafile_data= target_name, hierarchical_labels_one=hierarchy,hierarchical_labels_two=hierarchy_two, total_level=3,device=device)



    for epoch in range(epoch_num):
        model.train()
        for batchidx, (x, label1, label2, label3) in enumerate(tqdm(train_loader)):
            x, label1, label2, label3 = x.to(device), label1.to(device), label2.to(device), label3.to(device)
            output1, output2, output3 = model(x)

            loss1 = criterion1(output1, label1)
            loss2 = criterion2(output2, label2)
            loss3 = criterion3(output3, label3)
            
            prediction=[output1,output2,output3]   
            dloss = HLN.calculate_dloss(prediction, [label1, label2,label3]) #depense loss
            #lloss = HLN.calculate_lloss(prediction, [label1, label2,label3]) # layer loss
            
            #loss = loss1 + loss2 + loss3
            loss = dloss + loss1 + loss2 + loss3
            train_loss_l = loss1 + loss2 + loss3
            train_loss_dloss = dloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(epoch+1, 'train_Total_loss:', loss.item())
        print(epoch+1, 'train_loss_l:', train_loss_l.item())
        print(epoch+1, 'train_loss_dloss:', train_loss_dloss.item())
        train_Loss.append(loss.item())

        if (epoch+1) % 1 == 0:
            model.eval()
            pred1_list, pred2_list, pred3_list = [], [], []
            label1_list, label2_list, label3_list = [], [], []
            with torch.no_grad():
                total_correct, total_correct1, total_correct2, total_correct3 = 0, 0, 0, 0
                total_num = 0
                for x, label1, label2, label3 in val_loader:
                    x, label1, label2, label3 = x.to(device), label1.to(device), label2.to(device), label3.to(device)
                    output1, output2, output3 = model(x)
                    pred1, pred2, pred3 = output1.argmax(dim=1), output2.argmax(dim=1), output3.argmax(dim=1)
                    
                    loss1 = criterion1(output1, label1)
                    loss2 = criterion2(output2, label2)
                    loss3 = criterion3(output3, label3)
                    prediction = [output1,output2,output3]
                    dloss = HLN.calculate_dloss(prediction, [label1, label2,label3])#depeense loss
                    test_Total_loss = loss1 + loss2 + loss3 + dloss
                    test_loss_l = loss1 + loss2 + loss3
                    test_loss_dloss = dloss

                    pred = torch.cat((pred1, pred2, pred3), dim=-1)
                    label = torch.cat((label1, label2, label3), dim=-1)
                    correct = float(torch.equal(pred, label))
                    correct1 = torch.eq(pred1, label1).float().sum().item()
                    correct2 = torch.eq(pred2, label2).float().sum().item()
                    correct3 = torch.eq(pred3, label3).float().sum().item()
                    
                    total_correct += correct
                    total_correct1 += correct1
                    total_correct2 += correct2
                    total_correct3 += correct3
                    total_num += x.size(0)

                    
                    pred1_list.append(pred1.item())
                    label1_list.append(label1.item())
                    pred2_list.append(pred2.item())
                    label2_list.append(label2.item())
                    pred3_list.append(pred3.item())
                    label3_list.append(label3.item())


                print(epoch+1, 'test_Total_loss:', test_Total_loss.item())
                print(epoch+1, 'test_loss_l:', test_loss_l.item())
                print(epoch+1, 'test_loss_dloss:', test_loss_dloss.item())
                test_Loss.append(test_Total_loss.item())

                acc = total_correct / total_num
                acc1 = total_correct1 / total_num
                acc2 = total_correct2 / total_num
                acc3 = total_correct3 / total_num
                acc_list.append(acc)
                print('*************************')
                print(epoch+1, 'test acc1:', acc1)
                print(epoch+1, 'test acc2:', acc2)
                print(epoch+1, 'test acc3:', acc3)
                print(epoch+1, 'test acc:', acc)
                print('*************************')
                
                print('class1 list: ', target1_name)
                print('class2 list: ', target2_name)
                print('class3 list: ', target3_name)
                
                cm1 = confusion_matrix(pred1_list, label1_list)
                cm2 = confusion_matrix(pred2_list, label2_list)
                cm3 = confusion_matrix(pred3_list, label3_list)
                print('*************************')
                print(cm1)
                print(cm2)
                print(cm3)
                print('*************************')

                if acc > acc_best:
                    acc_best = acc
                    best_epoch = epoch+1
                    best_out={
                        'best_epoch':best_epoch,
                        'acc_best':acc_best,
                        'cm1':cm1,
                        'cm2':cm2,
                        'cm3':cm3,
                    }

                    torch.save(model.state_dict(), modelsavename)#best.pkl
                    print('Save best statistics done:!'+str(epoch+1))

                torch.save(model.state_dict(), modelfinalname )#final.pkl
    df=pd.DataFrame()
    df['trainLoss']=train_Loss
    df['test_Loss']=test_Loss
    df['acc_list']=acc_list
    df.to_csv('training_samll.csv',index=False)
    print('********acc_best********:' , best_out['acc_best'])
    print('********acc_best********:' , best_out['best_epoch'])
    print('********cm********:',best_out['cm1'] )
    print('********cm********:',best_out['cm2'] )
    print('********cm********:',best_out['cm3'] )

if __name__ == '__main__':
    train()


