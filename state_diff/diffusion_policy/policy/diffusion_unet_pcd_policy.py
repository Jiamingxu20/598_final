from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Dict, OrderedDict

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmoc
from robomimic.models.obs_nets import ObservationGroupEncoder
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

import diffusion_policy.model.vision.crop_randomizer as dmvc
# import diffusion_policy.model.vision.pointcloud_provider as dmvp
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.resample import create_named_schedule_sampler, LossAwareSampler
from diffusion_policy.policy.base_image_policy import apply_conditioning_init
from diffusion_policy.model.common.nn import mean_flat


class DiffusionUnetPcdPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            feat_proc='none', # 'none', 'linear', 'mlp'
            pcd_latent_dim=256,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': [],
            'spatial': [],
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            elif type == 'depth':
                obs_config['depth'].append(key)
            elif type == 'spatial':
                if shape_meta['obs']['point_cloud']['info']['rob_pcd']:
                    obs_key_shapes[key][1] = obs_key_shapes[key][1] + shape_meta['obs']['point_cloud']['info']['N_per_link']*6
                if shape_meta['obs']['point_cloud']['info']['eef_pcd']:
                    obs_key_shapes[key][1] = obs_key_shapes[key][1] + shape_meta['obs']['point_cloud']['info']['N_eef']
                obs_config['spatial'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)
        # load model
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(config.observation.encoder)
        observation_group_shapes = OrderedDict()
        
        observation_group_shapes["obs"] = OrderedDict(obs_key_shapes)
        
        encoder_kwargs['spatial']["core_kwargs"]["output_dim"] = pcd_latent_dim
        
        # update arg for feature processing
        if feat_proc == 'none':
            encoder_kwargs['spatial']["core_kwargs"]["use_feats"] = False
            encoder_kwargs['spatial']["core_kwargs"]["preproc_feats"] = False
            encoder_kwargs['spatial']["core_kwargs"]["proj_feats"] = False
        elif feat_proc == 'linear':
            encoder_kwargs['spatial']["core_kwargs"]["use_feats"] = True
            encoder_kwargs['spatial']["core_kwargs"]["preproc_feats"] = False
            encoder_kwargs['spatial']["core_kwargs"]["proj_feats"] = True
        elif feat_proc == 'mlp':
            encoder_kwargs['spatial']["core_kwargs"]["use_feats"] = True
            encoder_kwargs['spatial']["core_kwargs"]["preproc_feats"] = True
            encoder_kwargs['spatial']["core_kwargs"]["proj_feats"] = False
        elif feat_proc == 'distill':
            encoder_kwargs['spatial']["core_kwargs"]["use_feats"] = True
            encoder_kwargs['spatial']["core_kwargs"]["preproc_feats"] = False
            encoder_kwargs['spatial']["core_kwargs"]["proj_feats"] = False
        else:
            raise ValueError(f"Unsupported feat_proc: {feat_proc}")
        encoder_kwargs['spatial']['obs_randomizer_class'] = 'PointCloudRandomizer' #rmoc.PointCloudRandomizer()
        encoder_kwargs['spatial']["core_kwargs"]["pn_net_cls"] = kwargs.get('pn_net_cls', 'PointNet2Encoder')
        
        # load model
        encoder = ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        obs_encoder = encoder.nets['obs']
                
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        # if eval_fixed_crop:
        #     replace_submodules(
        #         root_module=obs_encoder,
        #         predicate=lambda x: isinstance(x, rmoc.CropRandomizer),
        #         func=lambda x: dmvp.PointCloudRandomizer(
        #         )
        #     )

        # create diffusion model
        self.deno_state_only = kwargs.get('deno_state_only', False)

        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_feature_dim = obs_feature_dim
        
        input_dim = obs_feature_dim if self.deno_state_only else action_dim + obs_feature_dim 
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        model = ConditionalUnet1D(
            input_dim=input_dim,
            output_dim=input_dim*2 if kwargs.get('variance_type', None) in  ['learned', 'learned_range'] else input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.obs_as_impainting = kwargs.get('obs_as_impainting', True)
        self.diffuse_action = kwargs.get('diffuse_action', False)

        self.mask_generator = LowdimMaskGenerator(
            output_dim=action_dim, 
            obs_dim=0 if obs_as_global_cond or self.obs_as_impainting else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.num_agents = kwargs.get('num_agents', 1)
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.sample_temperature = kwargs.get('sample_temperature', 1.0)
        self.skip_inv_model = kwargs.get('skip_inv_model', False)
        self.vis_state_traj = kwargs.get('visualize_state_denoising', False)
        self.n_his_inv = kwargs.get('n_his_inv', 1)
        self.n_fut_inv = kwargs.get('n_fut_inv', 1)


        self.kwargs = kwargs
        self.schedule_sampler = None
        if self.kwargs.get('schedule_sampler', None) == 'loss-second-moment':
            self.schedule_sampler = create_named_schedule_sampler(self.kwargs.get('schedule_sampler', None) , self.noise_scheduler.config.num_train_timesteps ) 

        if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
            device = self.device
            # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
            self.betas = self.noise_scheduler.betas.to(device)
            self.alphas = self.noise_scheduler.alphas.to(device)
            self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = torch.cat((torch.tensor([1]).to(device)  , self.alphas_cumprod[:-1]))
            self.alphas_cumprod_next = torch.cat((self.alphas_cumprod[1:], torch.tensor([0]).to(device) ))

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

            # calculations for posterior q(x_{t-1} | x_t, x_0)
            self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            self.posterior_log_variance_clipped = torch.log(
                torch.cat((torch.tensor([self.posterior_variance[1]]).to(device), self.posterior_variance[1:]))
            )
            self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
            )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        self.freeze_encoder = False
        self.training_stage = kwargs.get('training_stage', 1) # for compatibility with workspace
        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        B, T, obs_feature_dim = condition_data.shape
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for idx, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
                # model_output: (B, T, C) -> (B, C, T)
                model_output = model_output.swapaxes(1,2)
                trajectory = trajectory.swapaxes(1,2)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                ).prev_sample
            # trajectory: (B, C, T) -> (B, T, C)
            if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
                trajectory = trajectory.swapaxes(1,2)

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1).float()
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_impainting and self.deno_state_only:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,:] = nobs_features
            cond_mask[:,:To,:] = True

        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        
        # stage 0: observation encoder and inverse dynamics model
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1).float()
            
            # if not self.diffuse_action:
            #     nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            #     cond_data = nobs_features
            #     trajectory = cond_data.detach()

        elif self.obs_as_impainting and self.deno_state_only:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = nobs_features
            trajectory = cond_data.detach()
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
            
        T = trajectory.shape[1]

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        if self.kwargs.get('schedule_sampler', None) == 'loss-second-moment':
            timesteps, weights = self.schedule_sampler.sample(bsz, trajectory.device)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask].float()

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        terms = {}
        if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
            B, C = noisy_trajectory.shape[0], noisy_trajectory.shape[2]
            assert pred.shape == (B, noisy_trajectory.shape[1], C * 2)
            model_output, model_var_values = torch.split(pred, C, dim=2)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=2)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, 
                r=frozen_out: r,
                x_start=trajectory,
                x_t=noisy_trajectory,
                t=timesteps,
                clip_denoised=False,
                model_kwargs={'local_cond': local_cond, 'global_cond': global_cond}
            )["output"]
            assert model_output.shape == noise.shape == trajectory.shape


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        if "vb" in terms:
            diffuse_loss = F.mse_loss(model_output, target, reduction='none')
        else:
            diffuse_loss = F.mse_loss(pred, target, reduction='none')
        diffuse_loss = diffuse_loss * loss_mask.type(diffuse_loss.dtype)

        if self.schedule_sampler != None and isinstance(self.schedule_sampler, LossAwareSampler):
            diffuse_loss = mean_flat(diffuse_loss)
            if "vb" in terms:
                diffuse_loss = diffuse_loss + terms["vb"]*self.kwargs.get('vb_factor', 0.01)  # TODO

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    timesteps, diffuse_loss.detach()
                )
            diffuse_loss = (diffuse_loss * weights).mean()
        else:
            diffuse_loss = reduce(diffuse_loss, 'b ... -> b (...)', 'mean')
            if "vb" in terms:
                diffuse_loss = mean_flat(diffuse_loss)
                diffuse_loss = diffuse_loss + terms["vb"]*self.kwargs.get('vb_factor', 0.01) # TODO

            diffuse_loss = diffuse_loss.mean()

        loss_dict = {
            'loss': diffuse_loss,
            'diffuse_loss': diffuse_loss,
        }
        return loss_dict
        
    def set_training_mode(self, mode):
        """
        Set the training mode of the model.
        
        Args:
            mode (str): Training mode, one of ['disable_state_prediction', 'disable_encoder', 'enable_all'].
        """
        if mode == 'train_encoder':
            pass
        elif mode == 'train_diffusion':
            self.training_stage = 1
            
            pass
        # elif mode == 'train_all':
        #     self.freeze_encoder = False
        #     self.training_stage = 2

        #     TorchUtils.unfreeze_model_parameters(self.model)
        #     TorchUtils.unfreeze_model_parameters(self.obs_encoder)
        #     TorchUtils.unfreeze_model_parameters(self.inv_model)

        else:
            raise ValueError(f"Unsupported mode: {mode}")
