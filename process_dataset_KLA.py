import os
import shutil
import pandas as pd


def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return

projectfile='KLA'
csvtotalname = 'KLA'
base_root=os.getcwd()
root=base_root+'/dataset/'+projectfile
dataset_root=root+'/row/'
train_root = root+'/train/'
test_root = root+'/test/'

makedirs(train_root)
makedirs(test_root)



def datasave(SAVEROOT,FINALDF,FILENAME):
    Data_name=FILENAME+'.csv' #檔名使用機台名稱
    his = list(filter(lambda x: x[0: len(Data_name)]==Data_name, os.listdir(SAVEROOT)))
    os.chdir(SAVEROOT)  
    if len(his)==1:
        Hisdata=pd.read_csv(FILENAME+'.csv')
        #Hisdata['DATETIME']=pd.to_datetime(Hisdata['DATETIME'])
        Condata=pd.concat([Hisdata,FINALDF],axis=0,ignore_index=False)
        #Condata.drop_duplicates(['fileName','FileName','dataNumber'],keep='first',inplace=True)
        Condata.to_csv(FILENAME+'.csv',index=False)
    else:
        FINALDF.to_csv(FILENAME+'.csv',index=False)

def imgDataClean(fileName,dataset,rowroot,dstroot,typeset,datalen):
    makedirs(dstroot+fileName+'/')
    for img in dataset:
        shutil.copyfile(rowroot+'/'+img, dstroot+fileName+'/'+str(fileName)+'_'+img)
        print(dstroot+str(fileName)+'_'+img)
        df=pd.DataFrame()
        df['FileName']=[fileName]
        df['dataNumber']=str(datalen)
        df['dataType']=typeset
        df['picName']=str(fileName)+'_'+img
        datasave(root,df,csvtotalname)

def DataLess(value,fileListData):
    df=pd.DataFrame()
    df['FileName']=[value]
    df['dataNumber']=len(fileListData)
    df['dataType']='NAN'
    df['picName']='NAN'
    datasave(root,df,csvtotalname)


def process_data(rootPic):
    ''' rootPic : rowdata來源( file/ filename)'''
    for i in os.listdir(rootPic):
        #try:
        print(os.listdir(rootPic+i))
        fileList = list(filter(lambda x: x[-4:]=='.jpg', os.listdir(rootPic+i)))

        #if len(fileList)>=80:
        if i in select_defect:
            print('------------Data:'+str(i))
            cutNumber = int(len(fileList)*0.9)
            trainData = fileList[0:cutNumber]
            testData = fileList[cutNumber:]
            imgDataClean(i,trainData,rootPic+i,train_root,'train',len(fileList))
            imgDataClean(i,testData,rootPic+i,test_root,'test',len(fileList))
            #else:
                #DataLess(i,fileList)
        else: #select fileLIst
            print('---------DataLess:'+str(i))
            DataLess(i,fileList)               
        #except:
            #print('------------Error:'+str(i))
            #print('error'+str(i))




if __name__ == '__main__':
    select_defect = ['I-Nothing','T-PE-Hole','T-AS-Residue','I-PE-Abnormal','T-M2-Particle','E-AS-Residue','P-AS-Residue',
'I-M2-Small-Hole','I-M2-Deformation','I-Oil-Like','T-AS-SiN-Hole','P-M2-Residue','I-Scratch','T-M1-Particle','P-M2-Open',
'T-PE-Residue','I-M1-Deformation','P-PE-Residue','I-AS-Hole','P-M1-Residue']
    process_data(dataset_root)
