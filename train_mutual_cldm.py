from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyMutualDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
if __name__ == "__main__":
    resume_path = 'lightning_logs/version_73/checkpoints/epoch=19-step=62379.ckpt'
    single_model_path = './models/single_control_epoch12.ckpt'
    first_stage_path = './models/kl_f8.ckpt'
    batch_size = 1
    logger_freq = 2000
    learning_rate = 1e-5
    sd_locked = True
    control_locked = False
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/mutual_cldm_v15.yaml').cpu()

    if resume_path is not None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    else:
        single_model_state_dict = load_state_dict(single_model_path, location='cpu')
        
        first_stage_dict = {}
        cond_stage_dict = {}
        control_stage_dict = {}
        unet_dict = {}

        
        for key in single_model_state_dict.keys():
            if "first_stage_model" in key:
                adapted_key = key.replace("first_stage_model.", "")
                if adapted_key in model.first_stage_model.state_dict().keys():
                    first_stage_dict[adapted_key] = single_model_state_dict[key]
            
            elif "cond_stage_model" in key:
                adapted_key = key.replace("cond_stage_model.", "")
                if adapted_key in model.cond_stage_model.state_dict().keys():
                    cond_stage_dict[adapted_key] = single_model_state_dict[key]
            
            elif "control_model" in key:
                adapted_key = key.replace("control_model.", "")
                if adapted_key in model.control_model.state_dict().keys():
                    control_stage_dict[adapted_key] = single_model_state_dict[key]
            
            elif "diffusion_model" in key:
                adapted_key = key.replace("model.diffusion_model.", "")
                if adapted_key in model.model.diffusion_model.unet_target.state_dict().keys():
                    unet_dict[adapted_key] = single_model_state_dict[key]
        
        model.first_stage_model.load_state_dict(first_stage_dict)
        model.cond_stage_model.load_state_dict(cond_stage_dict)
        model.control_model.load_state_dict(control_stage_dict)
        model.model.diffusion_model.unet_target.load_state_dict(unet_dict)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.control_locked = control_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = MyMutualDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=4, precision=32, callbacks=[logger])


    # Train!
    trainer.fit(model, dataloader)
