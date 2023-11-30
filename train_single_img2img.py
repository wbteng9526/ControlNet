from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


if __name__ == "__main__":
    # Configs
    first_stage_path = './models/kl_f8.ckpt'
    cond_stage_path = './models/kl_f8.ckpt'
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/single_ldm_v15.yaml').cpu()
    model.first_stage_model.load_state_dict(load_state_dict(first_stage_path, location='cpu'))
    model.cond_stage_model.load_state_dict(load_state_dict(cond_stage_path, location='cpu'))
    model.learning_rate = learning_rate

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=2, precision=32, callbacks=[logger])


    # Train!
    trainer.fit(model, dataloader)
