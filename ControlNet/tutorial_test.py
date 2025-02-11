import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
ckpt_path = './models/control_sd15_ini_crop1000_resize640_test.ckpt'


model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))

torch.save(model.state_dict(), './models/control_sd15_ini_crop1000_resize640_test.pth')