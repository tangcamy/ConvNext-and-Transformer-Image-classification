import os 
import sys
sys.path.append(os.getcwd())
import timm
import torch
import torch.nn as nn
import pandas as pd
from dataset_single import MyDataset

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from model.loss import CrossEntropyLabelSmooth
from model import convnext
from model import transformer
import math
import csv
from tqdm import tqdm

model_select = 'convnext'
train_dir = 'data/single/train'
test_dir = 'data/single/test'
num_classes1 = 2
epoch_num = 100
batch_size = 32
t=30 #warmup 10 , 20,30
n_t=0.5
lr_rate = 0.001

#cvd-2
target_name ={
'name':['CFAR_NG','CFAR_OK']
}

'''
#20
target_name = {
'name': ['I-Nothing','T-PE-Hole','T-AS-Residue','I-PE-Abnormal','T-M2-Particle','E-AS-Residue','P-AS-Residue',
'I-M2-Small-Hole','I-M2-Deformation','I-Oil-Like','T-AS-SiN-Hole','P-M2-Residue','I-Scratch','T-M1-Particle','P-M2-Open',
'T-PE-Residue','I-M1-Deformation','P-PE-Residue','I-AS-Hole','P-M1-Residue']
}
'''

def train(target_name):
    acc_best = 0
    train_Loss,test_Loss,acc_list=[],[],[]

    train_dataset = MyDataset(train_dir, 'train',target_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(test_dir, 'test',target_name)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    target_name = train_dataset.get_target_name()
    #print(target_name)

    print('****************')

    device = torch.device('cuda')
    if model_select =='convnext':
        model = convnext.Convnext_single(num_class=num_classes1).to(device)
    else:
        model = transformer.Transformer_single(num_class=num_classes1).to(device)

    parameters_1 = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters_1, lr=lr_rate, weight_decay=1e-5)
    lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  0.1  if n_t * (1+math.cos(math.pi*(epoch - t)/(epoch_num-t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(epoch_num-t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    criterion1 = CrossEntropyLabelSmooth(num_classes=num_classes1).to(device)

    for epoch in range(epoch_num):
        model.train()
        for batchidx, (x, label1) in enumerate(tqdm(train_loader)):
            x, label1 = x.to(device), label1.to(device)
            output1 = model(x)
            loss1 = criterion1(output1, label1)
            
            loss = loss1 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(epoch+1, 'train_loss:', loss.item())
        train_Loss.append(loss.item())

        if (epoch+1) % 1 == 0:
            model.eval()
            pred1_list = []
            label1_list= []
            with torch.no_grad():
                total_correct, total_correct1 = 0, 0
                total_num = 0
                for x, label1 in val_loader:
                    x, label1 = x.to(device), label1.to(device)
                    output1 = model(x)
                    pred1 = output1.argmax(dim=1)
                    
                    loss1 = criterion1(output1, label1)
                    test_loss= loss1 
                           
                    correct = float(torch.equal(pred1, label1))
                    correct1 = torch.eq(pred1, label1).float().sum().item()
                    
                    total_correct += correct
                    total_correct1 += correct1

                    total_num += x.size(0)
                    
                    pred1_list.append(pred1.item())
                    label1_list.append(label1.item())

                print(epoch+1, 'test_loss:', test_loss.item())
                test_Loss.append(test_loss.item())

                acc = total_correct / total_num
                acc1 = total_correct1 / total_num
                acc_list.append(acc1)

                print('*************************')
                print(epoch+1, 'test acc1:', acc1)
                print(epoch+1, 'test acc:', acc)
                print('*************************')
                
                print('class1 list: ', target_name)

                
                #cm1 = confusion_matrix(pred1.item(), label1.item())
                cm1 = confusion_matrix(pred1_list, label1_list)

                print('*************************')
                print(cm1)
                print('*************************')

                if acc > acc_best:
                    acc_best = acc
                    best_epoch = epoch+1
                    best_out={
                        'best_epoch':best_epoch,
                        'acc_best':acc_best,
                        'cm':cm1
                    }
        
                    torch.save(model.state_dict(), 'CFAR_V2_1133_82-30best_Single.pth')#best.pkl
                    print('Save best statistics done:!'+str(best_epoch))

                torch.save(model.state_dict(), 'CFAR_V2_1133_82-30final_Single.pth')#best.pkl

    df=pd.DataFrame()
    df['trainLoss']=train_Loss
    df['test_Loss']=test_Loss
    df['acc_list']=acc_list
    df.to_csv('training.csv',index=False)
    print('********acc_best********:' , best_out['acc_best'])
    print('********acc_best********:' , best_out['best_epoch'])
    print('********cm********:',best_out['cm'] )
    #---- picture acc -----# plot error
    

if __name__ == '__main__':
    train(target_name)

