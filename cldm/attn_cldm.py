import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, DiffusionWrapper
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler



class MutualAttentionLDM(LatentDiffusion):

    def __init__(self, mutual_key, mutual_cond_stage_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutual_key = mutual_key
        self.mutual_cond_stage_key = mutual_cond_stage_key

    @torch.no_grad()
    def get_input(self, batch, *args, **kwargs):
        x_t, c_t = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        x_r, c_r = super().get_input(batch, self.mutual_key, cond_key=self.mutual_cond_stage_key, *args, **kwargs)
        c_r = None # TODO add camera delta R and delta T like zero-123
        return x_t, x_r, dict(c_crossattn=[c_t]), dict(c_crossattn=[c_r])
    
    def shared_step(self, batch, *args, **kwargs):
        x_t, x_r, c_t, c_r = self.get_input(batch, *args, *kwargs)
        loss = self(x_t, x_r, c_t, c_r)
        return loss
    
    def forward(self, x_t, x_r, c_t, c_r, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x_t.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x_t, x_r, c_t, c_r, t, *args, **kwargs)
    
    def apply_model(self, x_t_noisy, x_r, t, cond_t, cond_r, return_ids=False):
        assert isinstance(cond_t, dict)
        assert isinstance(cond_r, dict)
        x_recon = self.model(x_t_noisy, x_r, t, **cond_t, **cond_r)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
    
    def p_losses(self, x_t_start, x_r, cond_t, cond_r, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_t_start))
        x_t_noisy = self.q_sample(x_start=x_t_start, t=t, noise=noise)
        model_output = self.apply_model(x_t_noisy, x_r, t, cond_t, cond_r)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_t_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_t_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
