if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.policy.diffusion_traj_unet_pcd_policy import DiffusionTrajUnetPcdPolicy
from diffusion_policy.policy.diffusion_unet_pcd_policy import DiffusionUnetPcdPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
# from diffusion_policy.dataset.real_pcd_dataset import RealPcdDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetPcdWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(cfg.training.seed)  # For multi-GPU setups
        np.random.seed(seed)
        random.seed(seed)

        
        # configure model
        self.model: DiffusionUnetPcdPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetPcdPolicy = None
        
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure dataset
        self.dataset: BaseImageDataset
        self.dataset = hydra.utils.instantiate(cfg.task.dataset)

        self.global_step = 0
        self.epoch = 0


    def run(self, rank=0, world_size=1):
 
        cfg = copy.deepcopy(self.cfg)
        print('Running training rank:', rank)
        
        # Add a variable to keep track of the best validation loss
        best_val_loss = float('inf')  # Initialize to a high value
        epochs_without_improvement = 0  # Count epochs without improvement in validation loss
        max_epochs_without_improvement = cfg.max_epochs_without_improvement  # Stop training if no improvement in 30 epochs
        epochs_waited_before_next_stage = cfg.epochs_waited_before_next_stage  # Wait for 10 epochs before moving to the next stage


        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)


        if cfg.debug:
            cfg.training.num_epochs = 2000
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 30
            cfg.training.rollout_every = 2
            cfg.training.checkpoint_every = cfg.training.rollout_every
            cfg.training.val_every = cfg.training.rollout_every
            cfg.training.sample_every = cfg.training.rollout_every
            max_epochs_without_improvement = 2
            epochs_waited_before_next_stage = 2
            cfg.batch_size = 8
            cfg.dataloader.batch_size = cfg.batch_size
            cfg.val_dataloader.batch_size = cfg.batch_size
            cfg.task.dataset.max_train_episodes = 1
            cfg.task.dataset.val_ratio = 0.0001
            
        assert isinstance(self.dataset, BaseImageDataset)

        if world_size > 1:
            sampler = DistributedSampler(self.dataset)
            cfg.dataloader.shuffle = False
        else:
            sampler = None
            # cfg.dataloader.shuffle = True

        train_dataloader = DataLoader(
            self.dataset, 
            sampler=sampler,
            **cfg.dataloader
        )

        # if world_size > 1:
        #     train_dataloader = DataLoader(self.dataset, sampler=sampler, **cfg.dataloader)
        # else:
        #     train_dataloader = DataLoader(self.dataset, **cfg.dataloader)
        normalizer = self.dataset.get_normalizer()

        # configure validation dataset
        val_dataset = self.dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)


        if  rank == 0:
            # configure env
            env_runner: BaseImageRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir,
                debug=cfg.debug)
            assert isinstance(env_runner, BaseImageRunner)
            
            # configure logging
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                settings=wandb.Settings(_service_wait=300),
                **cfg.logging
            )
            # wandb.config.update(
            #     {
            #         "output_dir": self.output_dir,
            #     },
            #     allow_val_change=True,
            # )

            # configure checkpoint
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )
        else:
            wandb_run = None

        # device transfer
        if world_size > 1:
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(cfg.training.device)
            
        if world_size>1:
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)
            underlying_model = self.model.module
        else:
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            underlying_model = self.model
                
        if underlying_model.training_stage == 0:
            underlying_model.set_training_mode('train_encoder')
        elif underlying_model.training_stage == 1:
            underlying_model.set_training_mode('train_diffusion')
        else:
            raise ValueError(f"Unsupported training stage: {underlying_model.training_stage}")

        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        if rank == 0 or world_size == 1:
            log_path = os.path.join(self.output_dir, 'logs.json.txt')
        else:
            log_path = None
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this self.epoch ==========
                train_losses, train_diffuse_losses, train_inv_losses = [], [], []
                train_losses_dict = {}
                if world_size > 1:
                    train_dataloader.sampler.set_epoch(self.epoch)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss_dict = underlying_model.compute_loss(batch) 
                        raw_loss = raw_loss_dict if torch.is_tensor(raw_loss_dict) else raw_loss_dict['loss']
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(underlying_model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        if not torch.is_tensor(raw_loss_dict):
                            for key, value in raw_loss_dict.items():
                                if key == 'loss':
                                    continue
                                train_losses_dict[key] = train_losses_dict.get(key, [])
                                train_losses_dict[key].append(value.item() if isinstance(value, torch.Tensor) else value)

                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if rank == 0:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                for key, value in train_losses_dict.items():
                    if len(value) > 0:
                        step_log[key] = np.mean(value)
                step_log['freeze_encoder'] = int(underlying_model.freeze_encoder) 

                # ========= eval for this epoch ==========
                policy = underlying_model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and (rank == 0):
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and (rank == 0):
                    with torch.no_grad():
                        val_losses_dict = {}
                        val_losses = []
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_dict = underlying_model.compute_loss(batch)
                                loss = loss_dict if torch.is_tensor(loss_dict) else loss_dict['loss']
                                val_losses.append(loss)
                                if not torch.is_tensor(loss_dict):
                                    for key, value in loss_dict.items():
                                        if key == 'loss':
                                            continue
                                        val_losses_dict[f'val_{key}'] = val_losses_dict.get(key, [])
                                        val_losses_dict[f'val_{key}'].append(value.item() if isinstance(value, torch.Tensor) else value)


                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                        for key, value in val_losses_dict.items():
                            if len(value) > 0:
                                step_log[key] = np.mean(value)


                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            
                            # Update best validation loss and reset improvement counter
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                epochs_without_improvement = 0
                            else:
                                epochs_without_improvement += 1
                                
                            if 'traj' in cfg.policy._target_:
                                if epochs_without_improvement >= epochs_waited_before_next_stage:
                                    if underlying_model.training_stage == 0:
                                        print('Encoder training stage 0 completed. Moving to stage 1')
                                        underlying_model.set_training_mode('train_diffusion')
                                        # self.model.model_to_ddp(rank)
                                        epochs_without_improvement = 0
                                        best_val_loss = float('inf')

                                    # elif underlying_model.training_stage == 1:
                                    #     print('State prediction training stage 1 completed. Moving to stage 2')
                                    #     underlying_model.enable_all_model_training()
                                    #     # self.model.model_to_ddp(rank)
                                    #     epochs_without_improvement = 0
                                    #     best_val_loss = float('inf')
                                # if epochs_without_improvement >= epochs_waited_before_next_stage and :
                                #     self.model.disable_encoder_training()
                                #     epochs_without_improvement = 0
                                #     best_val_loss = float('inf')
                                # Check if the validation loss has not improved for a set number of epochs
                                if epochs_without_improvement >= max_epochs_without_improvement and underlying_model.training_stage == 1:
                                    print("Stopping training due to lack of improvement in validation performance.")
                                    break
                            else:
                                if epochs_without_improvement >= max_epochs_without_improvement :
                                    print("Stopping training due to lack of improvement in validation performance.")
                                    break


                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and (rank == 0):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        if 'traj' in cfg.policy._target_ and not cfg.policy.skip_inv_model:
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action[:,cfg.policy.n_his_inv-1:-cfg.policy.n_fut_inv])
                        else:
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action[:,:pred_action.shape[1]])

                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and (rank == 0):
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    metric_dict['stage'] = 1 if (underlying_model.freeze_encoder)  else 0
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if  rank == 0:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetPcdWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
