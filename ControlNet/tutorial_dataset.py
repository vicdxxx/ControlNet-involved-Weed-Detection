import json
import cv2
import numpy as np

from torch.utils.data import Dataset

dataname = 'SpottedSpurge_bbox' 

def letterbox(img, new_shape):
    shape = img.shape[:2]  # current shape [height, width]
    stride = 1
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    return img

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        with open(f'./training/{dataname}/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        source = cv2.imread(f'./training/{dataname}/' + source_filename)
        target = cv2.imread(f'./training/{dataname}/' + target_filename)

        h,w = source.shape[:2]
        # upper_size = 512
        upper_size = 640
        # upper_size = 768
        # upper_size = 1024
        aspect = h/w
        if h > w:
            source = cv2.resize(source, (int(upper_size/aspect), upper_size))
            target = cv2.resize(target, (int(upper_size/aspect), upper_size))
        else:
            source = cv2.resize(source, (upper_size, int(upper_size*aspect)))
            target = cv2.resize(target, (upper_size, int(upper_size*aspect)))
        source = letterbox(source, upper_size)
        target = letterbox(target, upper_size)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class MyDatasetTest(Dataset):
    def __init__(self):
        self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        with open(f'./training/{dataname}/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        source = cv2.imread(f'./training/{dataname}/' + source_filename)
        target = cv2.imread(f'./training/{dataname}/' + target_filename)

        h,w = source.shape[:2]
        # upper_size = 512
        upper_size = 640
        # upper_size = 768
        # upper_size = 1024
        aspect = h/w
        if h > w:
            source = cv2.resize(source, (int(upper_size/aspect), upper_size))
            target = cv2.resize(target, (int(upper_size/aspect), upper_size))
        else:
            source = cv2.resize(source, (upper_size, int(upper_size*aspect)))
            target = cv2.resize(target, (upper_size, int(upper_size*aspect)))
        source = letterbox(source, upper_size)
        # target = letterbox(target, upper_size)
        target = letterbox(np.zeros_like(target), upper_size)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
