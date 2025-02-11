from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset, MyDatasetTest
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
# resume_path = './models/control_sd15_ini.ckpt'
resume_path = r'D:\BoyangDeng\StableDiffusion\ControlNet\models\control_sd15_ini.ckpt'
# resume_path = r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Purslane\version_0\checkpoints\epoch=25-step=6733.ckpt'
batch_size = 4
# batch_size = 2
# batch_size = 1
logger_freq = 1000
# learning_rate = 0
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
# dataset = MyDatasetTest()

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, max_images=100)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger],accumulate_grad_batches=32)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], default_root_dir=r'D:\test', max_epochs=10)
# max_epochs 30 50
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs=50)
# trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], max_steps=30)


# Train
trainer.fit(model, dataloader)
# trainer.validate(model, dataloader)
# trainer.test(dataloaders=dataloader, model=model)
