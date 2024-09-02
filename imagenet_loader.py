from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import json
class imagenet_dataset(Dataset):
    def __init__(self, root_dir: str, label_dir: str, transform=None):
        """
        初始化数据集类
        :param root_dir: 训练数据集的根目录
        :param transform: 数据增强的变换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classify = sorted(os.listdir(root_dir))  # 获取所有类别的文件夹名
        self.len = 0
        self.data = []
        self.classify_idx = np.zeros(len(self.classify), dtype=np.int32)
        for i, cla in enumerate(self.classify):
            img_path = os.path.join(root_dir, cla)
            for img in os.listdir(img_path):
                route = os.path.join(img_path, img)
                self.len += 1
                self.data.append(route)
            self.classify_idx[i] = self.len
        self.data = np.array(self.data)
        # print(type(self.data[0]))
        self.load_json(label_dir)

    def load_json(self, label_dir:str):
        with open(label_dir) as f:
            data = f.read()
            load = json.loads(data)
            self.label = np.zeros(len(self.classify), dtype=object)
            for i, cla in enumerate(self.classify):
                self.label[i] = load[cla]

    def get_label(self, index):
        return self.label[index]

    def __getitem__(self, index):
        """
        获取一张图片的数据和标签
        :param index: 图片的索引
        :return: 图片数据和标签
        """
        # 二分搜索确定 label_idx(大于index的最小整数)
        low, high = 0, len(self.classify_idx) - 1
        while low < high:
            mid = (low + high) // 2
            if self.classify_idx[mid] > index:
                high = mid
            else:
                low = mid + 1
        label_idx = low
        # 打开图片
        img = Image.open(self.data[index]).convert('RGB') # 将单通道的图转为三通道
        if self.transform:
            img = self.transform(img)
        return img, label_idx

    def __len__(self):
        return self.len

# import torchvision.transforms as transforms

# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),  # 期望的输入尺寸
#         transforms.ToTensor(),
#         transforms.Normalize(0.5, 0.5),
#     ]
# )
# train_data = imagenet_dataset(
#     root_dir="data/imagenet100/train",
#     label_dir="data/imagenet100/Labels_100.json",
#     transform=transform,
# )
# for d in train_data:
#     print(d[0].shape)
# print(train_data.get_label(0))
# print(train_data[0])
# print(train_data[128750])
