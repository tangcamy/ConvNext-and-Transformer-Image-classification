# 專案說明
影像分類模型，研究因素（特徵是否共享/是否為階層式結構 Hierarchical）
使用的模型有ConvNext ,transformer

# paper 
- [Paper with code:Convnext](https://paperswithcode.com/method/convnext)
- [transformer computer -version](https://www.edge-ai-vision.com/2022/05/transformers-in-computer-vision/)
- [Paper with code : Deep Hierarchical Classification for Category Prediction in E-commerce System ](https://paperswithcode.com/paper/deep-hierarchical-classification-for-category)


# 資料夾說明
- cam_pic(儲存熱力圖的資料夾)
- dataset :放原始資料的地方，請先在此資料底下創見自己專案名稱資料夾。
- data:
    - data/muti : 3個標籤lable data資料夾
    - data/single :單個標籤label data資料夾
- model(依據訓練不同條件選擇模型種類 & LossFunction)
    - convnext.py : Convnext_single,Convnext_muti,Convnext_sharefeature
    - transformer.py : Transformer_single,Transformer_muti,Transformer_sharefeature
    - loss.py : 一個lable loss 計算。
    - hierarchical_loss.py:父子階層loss計算。目前3層結構)
    - level_dic.py:定義父子階層的字典。（目前3層結構)

# Train 步驟
## 〔　1. 資料前處理:處理後放在dataset，需複製於data〕
    - 單一輸出範例程式：process_dataset_KLA.py  
        - 輸出的資料夾(train&test)放置/data/single/
    - 多輸出(目前3個）範例程式：process_dataset_FMA.py
        - 輸出的資料夾(train&test)放置/data/muti/

## 〔　2. 相關設定：level_dict.py　〕/model file裡面
    - 設定相關階層資料。（多層才需要）

## 〔　3. DataLoader： 預設 input size224*224　〕
    - dataset_single.py :單個輸出，
    - dataset_muti.py :3個輸出。

## 〔　4.model選擇:需改程式階層數量　〕
    - 常見一般分類 （單輸出）Train_single.py: Convnext_single / Transformer_single
    - 常見一般分類（多輸出）Train_Feature_LayerLosspy : Convnext_muti / Transformer_muti
    - 階層式框架（特徵獨立）Train_Feature_Hierarchicalloss.py : Convnext_muti / Transformer_muti
    - 階層式框架（特徵共享）Train_ShareFeature_Hierarchicalloss.py : Convnext_sharefeature / Transformer_sharefeature
    

# Inferenct 步驟
## 〔　1. 預測 〕
    - detect.py ：模型記得選擇，裡面已修改成包含熱力圖。
## 〔　2. 熱力圖：備用 〕
    - cam_mutiPic.py :整個資料夾預測，目前用在ConvNext模型。
    - cam_onepic.py :單張照片預測，目前用在ConvNext模型。
