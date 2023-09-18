from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_dir, phase,target_names):
        self.transform_train = transforms.Compose([
                        transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
                                            
        self.transform_test = transforms.Compose([
                        transforms.Resize((224, 224), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

        self.target_name = target_names['name']
        self.phase = phase
        if self.phase == 'pred':
            self.image_path = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        else:
            self.image_path = []
            self.target = []

            class_list = os.listdir(data_dir)
            for i, class_ in enumerate(class_list):
                name = class_
                   
                data_list = [os.path.join(data_dir, class_, i) for i in os.listdir(os.path.join(data_dir, class_))]
                print(data_list)
                self.image_path.extend(data_list)
                self.target.extend([self.target_name.index(name)]*len(data_list))
                
                

    def get_target_name(self):
        return self.target_name

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):
        if self.phase == 'pred':
            img_path= self.image_path[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform_test(img)
            return img,img_path
        else:
            img_path, label = self.image_path[index], self.target[index]
            img = Image.open(img_path).convert('RGB')
            if self.phase == 'train':
                img = self.transform_train(img)
            elif self.phase == 'test':
                img = self.transform_test(img)

            return img, label
