import os
import shutil
import pandas as pd


def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return

projectfile='FMA'#'FMA'
csvtotalname = 'FMA_Process_less'
base_root='/home/orin/L5C_CellFMA/DeepHieral/'
root=base_root+'dataset/'+projectfile
dataset_root=root+'/row/'
train_root = root+'/train/'
test_root = root+'/test/'

makedirs(train_root)
makedirs(test_root)



def NameListCheck(value):
    a=[value]
    a=a[0].split('@')
    FileNmae = a[0]
    LocationFlag = a[1]
    DefectLocate = a[2]
    site = a[3]
    return FileNmae,LocationFlag,DefectLocate,site

def imgDataClean(fileName,dataset,rowroot,dstroot,typeset,datalen):
    makedirs(dstroot+fileName+'/')
    for img in dataset:
        shutil.copyfile(rowroot+'/'+img, dstroot+fileName+'/'+str(fileName)+'_'+img)
        print(dstroot+str(fileName)+'_'+img)
        a=[fileName]
        a=a[0].split('@')
        df=pd.DataFrame()
        df['FileName']=[fileName]
        df['dataNumber']=str(datalen)
        df['locationFlag']=a[1]
        df['dataType']=typeset
        df['picName']=str(fileName)+'_'+img
        datasave(root,df,csvtotalname)

def DataLess(value,fileListData):
    df=pd.DataFrame()
    df['FileName']=[value]
    df['dataNumber']=len(fileListData)
    df['locationFlag']='NAN'
    df['dataType']='NAN'
    df['picName']='NAN'
    datasave(root,df,csvtotalname)

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

def process_data(rootPic):
    ''' rootPic : rowdata來源( file/ filename)'''
    for i in os.listdir(rootPic):
        #i='GLASS CULLET@UCT@TFT@Cell'
        fileList = list(filter(lambda x: x[-4:]=='.jpg', os.listdir(rootPic+i)))
        if len(fileList)>=80:
            print('------------Data:'+str(i))
            cutNumber = int(len(fileList)*0.9)
            trainData = fileList[0:cutNumber]
            testData = fileList[cutNumber:]
            FileNameCheck = NameListCheck(i)
            if FileNameCheck[2] == 'CF' or FileNameCheck[2] == 'TFT':
                imgDataClean(i,trainData,rootPic+i,train_root,'train',len(fileList))
                imgDataClean(i,testData,rootPic+i,test_root,'test',len(fileList))
            else:
                DataLess(i,fileList)
        else:
            #print('---------DataLess:'+str(i))
            DataLess(i,fileList)               



if __name__ == '__main__':
    process_data(dataset_root)
