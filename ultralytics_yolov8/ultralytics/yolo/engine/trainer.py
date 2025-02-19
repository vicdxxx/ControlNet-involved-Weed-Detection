# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Train a model on a dataset

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""
import os
from random import Random
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import copy
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm

from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer, is_parallel)
import config_da as cfg_da
import config_ss as cfg_ss
import da_I3Net_module as da_I3Net
import config as cfg_tot

if cfg_ss.use_semisupervised:
    from semisupervised.ss_config import (
        LEARNING_RATE,
        DECAY,
        THRESHOLDS_TO_TEST,
        THRESHOLD_OVERESTIMATE,
        FLIP_PROBABILITY,
        BATCH_SIZE,
        INSTANCES_PER_UPDATE,
        NMS_IOU_THRESHOLD,
    )
    from semisupervised.pytorch_augmentations import (
        horizontal_flip,
        vertical_flip,
        deterministic_horizontal_flip,
        deterministic_vertical_flip,
        bbox_hflip,
        bbox_vflip,
    )

class BaseTrainer:
    """
    BaseTrainer

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        check_resume (method): Method to check if training should be resumed from a saved checkpoint.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to last checkpoint.
        best (Path): Path to best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataloaders.
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e
            print(traceback.format_exc())
        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """
        Appends the given callback.
        """
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """
        Overrides the existing callbacks with the given callback.
        """
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        # Allow device='', device=None on Multi-GPU systems to default to device=0
        if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting rect=False")
                self.args.rect = False
            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'Running DDP command {cmd}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(traceback.format_exc())
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _setup_ddp(self, world_size):
        torch.cuda.set_device(RANK)
        self.device = torch.device('cuda', RANK)
        LOGGER.info(f'DDP settings: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo', rank=RANK, world_size=world_size)

    def load_pretained_model(self):
        model, weights = self.args.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg=ckpt['model'].yaml
        else:
            cfg=model
        loaded_model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        loaded_model.names = self.data['names']
        return loaded_model

    def _setup_train(self, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        if cfg_da.use_domain_adaptation:
            cfg_da.da_info['td_cls_num'] = self.model.nc

            self.L2Norm = da_I3Net.L2Norm(cfg_da.da_info['channel_num'] , cfg_da.da_info['td_cls_num'])
            self.netD_pixel = da_I3Net.netD_pixel()
            self.netD = da_I3Net.netD()
            self.conv_gcr = da_I3Net.net_gcr_simple(cfg_da.da_info['channel_num'] , cfg_da.da_info['td_cls_num'])
            self.RandomLayer = da_I3Net.RandomLayer([cfg_da.da_info['channel_num'] , cfg_da.da_info['td_cls_num']], 1024)

            self.old_state = torch.zeros(self.model.nc)
            self.new_state = torch.zeros(self.model.nc)

            if cfg_da.da_info['model_L2Norm'] is not None:
                assert cfg_da.da_info['model_L2Norm'] is not None
                assert cfg_da.da_info['model_netD_pixel'] is not None
                assert cfg_da.da_info['model_netD'] is not None
                assert cfg_da.da_info['model_conv_gcr'] is not None
                assert cfg_da.da_info['model_RandomLayer'] is not None
                #def intersect_dicts(da, db, exclude=()):
                #    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
                #    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}
                #csd = intersect_dicts(csd, cfg_da.da_info['model_L2Norm'].state_dict())  # intersect
                #self.L2Norm.load_state_dict(cfg_da.da_info['model_L2Norm'], strict=False)
                self.L2Norm = cfg_da.da_info['model_L2Norm']
                self.netD_pixel = cfg_da.da_info['model_netD_pixel']
                self.netD = cfg_da.da_info['model_netD']
                self.conv_gcr = cfg_da.da_info['model_conv_gcr']
                self.RandomLayer = cfg_da.da_info['model_RandomLayer']

            if cfg_da.da_info['new_state'] is not None:
                self.old_state = cfg_da.da_info['old_state']
                self.new_state = cfg_da.da_info['new_state']
            # no bg class

            self.old_state = self.old_state.to(self.device)
            self.new_state = self.new_state.to(self.device)

            self.L2Norm = self.L2Norm.to(self.device)
            self.netD_pixel = self.netD_pixel.to(self.device)
            self.netD = self.netD.to(self.device)
            self.conv_gcr = self.conv_gcr.to(self.device)
            self.RandomLayer = self.RandomLayer.to(self.device)

            self.softmax = nn.Softmax(dim=-1)
            self.pa_list = cfg_da.da_info['pa_list'] 
            self.fea_lists = [[torch.tensor([]).to(self.device) for _ in range(cfg_da.da_info['td_cls_num'])] for _ in range(len(self.pa_list))]
            self.fea_lists_t = [[torch.tensor([]).to(self.device) for _ in range(cfg_da.da_info['td_cls_num'])] for _ in range(len(self.pa_list))]

        # Check AMP: Automatic Mixed Precision
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            try:
                callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
                self.amp = torch.tensor(check_amp(self.model), device=self.device)
                if cfg_da.use_domain_adaptation:
                    self.amp_L2Norm = torch.tensor(check_amp(self.L2Norm), device=self.device)
                    self.amp_netD_pixel = torch.tensor(check_amp(self.netD_pixel), device=self.device)
                    self.amp_netD = torch.tensor(check_amp(self.netD), device=self.device)
                    self.amp_conv_gcr = torch.tensor(check_amp(self.conv_gcr), device=self.device)
                    self.amp_RandomLayer = torch.tensor(check_amp(self.RandomLayer), device=self.device)
                    self.amp = self.amp and self.amp_L2Norm and self.amp_netD_pixel and self.amp_netD and self.amp_conv_gcr and self.amp_RandomLayer
                callbacks.default_callbacks = callbacks_backup  # restore callbacks
            except Exception as e:
                self.amp = torch.tensor(False).to(self.device)
                print(e)
                print(traceback.format_exc())
            print('self.amp:', self.amp)

        if RANK > -1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])

        if cfg_ss.use_semisupervised:
            if cfg_ss.use_semi_ema_as_teacher:
                teacher_model = None
            else:
                teacher_model = self.load_pretained_model()
            from semisupervised.semi_supervised import SemiSupervised
            self.ss_framework = SemiSupervised(
                model=self.model,
                teacher_model=teacher_model,
                model_base='YOLOv8',
                num_classes=len(cfg_ss.CLASSNAME_TO_IDX),
            )
            self.ss_framework.device = self.device
            self.ss_framework.preprocess_batch = self.preprocess_batch

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay

        if cfg_da.use_domain_adaptation:
            #model_add_ad = nn.ModuleList([self.model, self.L2Norm, self.netD_pixel, self.netD, self.conv_gcr, self.RandomLayer])
            model_add_ad = [self.model, self.L2Norm, self.netD_pixel, self.netD, self.RandomLayer, self.conv_gcr]
            self.optimizer = self.build_optimizer(model=model_add_ad,
                                                  name=self.args.optimizer,
                                                  lr=self.args.lr0,
                                                  momentum=self.args.momentum,
                                                  decay=weight_decay)
        else:
            self.optimizer = self.build_optimizer(model=self.model,
                                                  name=self.args.optimizer,
                                                  lr=self.args.lr0,
                                                  momentum=self.args.momentum,
                                                  decay=weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # dataloaders
        batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        self.loss_names_train = 'box_loss', 'cls_loss', 'dfl_loss'
        if cfg_da.use_domain_adaptation:
            self.da_td_loader = self.get_dataloader(cfg_da.da_info['td_dataset_train'], batch_size=batch_size, rank=RANK, mode='train', domain_adaptation=True)
        elif cfg_ss.use_semisupervised:
            self.ss_td_loader = self.get_dataloader(cfg_ss.ss_info['td_dataset_train'], batch_size=batch_size, rank=RANK, mode='train', domain_adaptation=False, close_mosaic=False, need_label=False)
            self.ss_td_loader.dataset.labelled_dataset = self.train_loader.dataset
            self.ss_td_loader.dataset.build_transform_for_SSOD(hyp=self.args)

        if RANK in (-1, 0):
            # batch_size = batch_size * 2 / batch_size
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size, rank=-1, mode='val')
            self.validator = self.get_validator(self.args)

            if cfg_ss.use_semisupervised:
                self.ss_framework.validator = self.validator

            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
            if self.args.plots and not self.args.v5loader:
                self.plot_training_labels()
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        if cfg_da.use_domain_adaptation or cfg_ss.use_semisupervised:
            nw = max(round(self.args.warmup_epochs * nb), 0)  # number of warmup iterations
        else:
            # nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
            nw = max(round(self.args.warmup_epochs * nb), 0)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        img_per_epoch = nb * self.batch_size

        #self.metrics, self.fitness = self.validate()

        if cfg_da.use_domain_adaptation:
            da_td_batch_iterator = iter(self.da_td_loader)

        if cfg_ss.use_semisupervised:
            print('cfg_ss.start_epoch:', cfg_ss.start_epoch)
            ss_td_batch_iterator = None

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()

            loss_names_train = self.loss_names_train
            if cfg_ss.use_semisupervised:
                if epoch < cfg_ss.start_epoch: 
                    cfg_ss.use_semisupervised_in_loop = False
                else:
                    if cfg_ss.use_semi_ema_as_teacher:
                        if cfg_ss.reinitialize_model_each_epoch:
                            print('SSOD: reinitialize_model_each_epoch')
                            msd = self.model.module.state_dict() if is_parallel(self.model) else self.model.state_dict()  # model state_dict
                            for k, v in self.ema.ema.state_dict().items():
                                if v.dtype.is_floating_point:
                                    msd[k] = v
                        if cfg_ss.reinitialize_semi_ema_each_epoch:
                            print('SSOD: reinitialize_semi_ema_each_epoch')
                            self.ss_framework.init_teacher_model_by_efficient_teacher(self.ema, self.epochs, cfg_ss.start_epoch)
                        else:
                            if epoch == cfg_ss.start_epoch:
                                self.ss_framework.init_teacher_model_by_efficient_teacher(self.ema, self.epochs, cfg_ss.start_epoch)
                    if ss_td_batch_iterator is None:
                        print('SSOD: init in cfg_ss.start_epoch')
                        ss_td_batch_iterator = iter(self.ss_td_loader)
                        """
                        only do this obtain 0.6%+ mAP@50:95?
                        if add things in use_semisupervised_in_loop, then the acc will drop?
                        """
                        self.ss_framework.on_train_start(self.ss_td_loader)
                        ss_update_every = INSTANCES_PER_UPDATE / self.batch_size
                        print('ss_update_every:', ss_update_every)
                        if cfg_ss.use_strong_aug_for_student:
                            from ultralytics.yolo.data.augment import Mosaic, MixUp, RandomPerspective, RandomHSV, Albumentations
                            model_randomperspective = RandomPerspective(degrees=0, translate=0.1, scale=0.5)
                            model_albumentations = Albumentations(p=1.0)
                            model_randomHSV = RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4)
                            if cfg_ss.use_mosaic_and_mixup_for_student:
                                model_mosaic = Mosaic(dataset=None, imgsz=cfg_ss.image_size, p=1.0)
                                model_mixup = MixUp(dataset=None, pre_transform=None, p=1.0)
                    cfg_ss.use_semisupervised_in_loop = True
                    loss_names_train = self.loss_names_train + ('t_box_loss', 't_cls_loss', 't_dfl_loss')
                    print('cfg_ss.use_semisupervised_in_loop:', cfg_ss.use_semisupervised_in_loop)
            print('loss_names_train:', loss_names_train)
            
            if cfg_da.use_domain_adaptation:
                self.L2Norm.train()
                self.netD_pixel.train()
                self.netD.train()
                self.conv_gcr.train()
                self.RandomLayer.train()
                loss_names_train = self.loss_names_train + ('dloss_l', 'dloss_l_t', 'dloss_g', 'dloss_g_t', 'loss_gf', 'loss_gcr', 'loss_gpa', 'loss_kl_tot')

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string(loss_names_train))
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()

            for i, batch in pbar:
                if cfg_tot.is_debug:
                    # ensure ss_update_every work
                    if i >= cfg_tot.step_num_per_epoch:
                        break
                self.run_callbacks('on_train_batch_start')
                
                if cfg_ss.use_semisupervised_in_loop and not cfg_ss.use_semi_ema_as_teacher:
                    if ni >= nw: # after warmup
                        if i > 0 and i % ss_update_every == 0:
                            self.ss_framework.model = self.model
                            self.ss_framework.update_teacher()
                        
                # Warmup
                ni = i + nb * epoch
                if ni < nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        if cfg_da.use_domain_adaptation:
                            if j > 2:
                                x['lr'] = 0
                                #break
                        x['lr'] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                if cfg_da.use_domain_adaptation:
                    try:
                        batch_t = next(da_td_batch_iterator)
                    except StopIteration:
                        print('StopIteration: da_td_batch_iterator, reinitiateing')
                        da_td_batch_iterator = iter(self.da_td_loader)
                        batch_t = next(da_td_batch_iterator)
                
                if cfg_ss.use_semisupervised_in_loop:
                    try:
                        batch_unlabelled = next(ss_td_batch_iterator)
                    except StopIteration:
                        print('StopIteration: da_td_batch_iterator, reinitiateing')
                        ss_td_batch_iterator = iter(self.ss_td_loader)
                        batch_unlabelled = next(ss_td_batch_iterator)

                # Initialize
                if cfg_da.use_domain_adaptation:
                    sources = list()
                    loc = list()
                    conf = list()
                    fea_lists = []
                    pre_lists = []

                    sources_t = list()
                    loc_t = list()
                    conf_t = list()
                    fea_lists_t = []
                    pre_lists_t = []

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    #print(batch['im_file'][0])
                    #cfg_tot.show_image_with_bbox(batch['img'][0], rgb2bgr=1)
                    preds = self.model(batch['img'])
                    # if cfg_tot.is_debug:
                    #     pred_final = self.validator.postprocess(preds, conf=cfg_ss.teacher_lowest_postprocess_conf_t, iou=0.5)[0]
                    # dataset A operations
                    if cfg_da.use_domain_adaptation:
                        # [4, 6, 9, 12, 15, 18, 21]
                        preds, itermediate_outputs = preds
                        domain_l = self.netD_pixel(da_I3Net.grad_reverse(itermediate_outputs[4]))
                        domain_g = self.netD(da_I3Net.grad_reverse(itermediate_outputs[15]))
                        gcr_pre = self.conv_gcr(itermediate_outputs[15])
                        feat1 = itermediate_outputs[15]

                        source = self.L2Norm(itermediate_outputs[15])
                        sources.append(source)
                        sources.append(itermediate_outputs[18])
                        sources.append(itermediate_outputs[21])

                        loc.append(preds[0][:, :-self.model.nc])
                        loc.append(preds[1][:, :-self.model.nc])
                        loc.append(preds[2][:, :-self.model.nc])

                        conf.append(preds[0][:, -self.model.nc:])
                        conf.append(preds[1][:, -self.model.nc:])
                        conf.append(preds[2][:, -self.model.nc:])

                        for i_src, src in enumerate(sources):
                            loc[i_src] = loc[i_src].permute(0, 2, 3, 1).contiguous()
                            conf[i_src] = conf[i_src].permute(0, 2, 3, 1).contiguous()
                            if i_src > 0:
                                fea_list = da_I3Net.get_fea_list(src.permute(0, 2, 3, 1).contiguous(), conf[i_src], self.model.nc)
                                fea_lists.append(fea_list)
                                pre_lists.append(conf[i_src])
                            if i_src == 0:
                                feat2 = conf[i_src]
                                g_feat = da_I3Net.get_feature_vector(feat1, feat2.detach(), self.softmax, self.RandomLayer, self.model.nc)
                        self.fea_lists = da_I3Net.Moving_average(fea_lists, self.fea_lists)
                        loss_kl = torch.tensor(0)

                    # dataset B operations
                    if cfg_da.use_domain_adaptation:
                        batch_t = self.preprocess_batch(batch_t)
                        #print(batch_t['im_file'][0])
                        #cfg_tot.show_image_with_bbox(batch_t['img'][0], rgb2bgr=1)
                        preds_t = self.model(batch_t['img'])
                        preds_t, itermediate_outputs_t = preds_t
                        domain_l_t = self.netD_pixel(da_I3Net.grad_reverse(itermediate_outputs_t[4]))
                        domain_g_t = self.netD(da_I3Net.grad_reverse(itermediate_outputs_t[15]))
                        gcr_pre_t = self.conv_gcr(itermediate_outputs_t[15])
                        feat1_t = itermediate_outputs_t[15]

                        source_t = self.L2Norm(itermediate_outputs_t[15])
                        sources_t.append(source_t)
                        sources_t.append(itermediate_outputs_t[18])
                        sources_t.append(itermediate_outputs_t[21])

                        loc_t.append(preds_t[0][:, :-self.model.nc])
                        loc_t.append(preds_t[1][:, :-self.model.nc])
                        loc_t.append(preds_t[2][:, :-self.model.nc])

                        conf_t.append(preds_t[0][:, -self.model.nc:])
                        conf_t.append(preds_t[1][:, -self.model.nc:])
                        conf_t.append(preds_t[2][:, -self.model.nc:])

                        for i_src_t, src_t in enumerate(sources_t):
                            loc_t[i_src_t] = loc_t[i_src_t].permute(0, 2, 3, 1).contiguous()
                            conf_t[i_src_t] = conf_t[i_src_t].permute(0, 2, 3, 1).contiguous()
                            if i_src_t > 0:
                                fea_list_t = da_I3Net.get_fea_list(src_t.permute(0, 2, 3, 1).contiguous(), conf_t[i_src_t], self.model.nc)
                                fea_lists_t.append(fea_list_t)
                                pre_lists_t.append(conf_t[i_src_t])
                            if i_src_t == 0:
                                feat2_t = conf_t[i_src_t]
                                g_feat_t = da_I3Net.get_feature_vector(feat1_t, feat2_t.detach(), self.softmax, self.RandomLayer, self.model.nc)
                        self.fea_lists_t = da_I3Net.Moving_average(fea_lists_t, self.fea_lists_t)
                        loss_kl_t = da_I3Net.get_kl_loss(pre_lists_t, self.softmax, self.model.nc)
                
                    if cfg_ss.use_semisupervised_in_loop and batch_unlabelled is not None:
                        assert self.ss_framework.teacher is not None
                        self.ss_framework.teacher.eval()
                        batch_unlabelled = self.preprocess_batch(batch_unlabelled)
                        unlabelled_images = batch_unlabelled['img']
                        with torch.no_grad():
                            images_ss, targets_ss = [], []
                            preds_teacher = []
                            for img in unlabelled_images:
                                if cfg_ss.use_ema_only_for_teacher_predictor:
                                    teacher_preds = self.ss_framework.weakly_ensembled_teacher_forward(img, detector=self.ema.ema)
                                else:
                                    teacher_preds = self.ss_framework.weakly_ensembled_teacher_forward(img)
                                preds_teacher.append(
                                    self.ss_framework._combine(teacher_preds, img.shape, self.ss_framework.thresholds,)
                                )
                            for img, pred in zip(unlabelled_images, preds_teacher):
                                img, pred = horizontal_flip(img, pred, FLIP_PROBABILITY)
                                img, pred = vertical_flip(img, pred, FLIP_PROBABILITY)
                                images_ss.append(img)
                                targets_ss.append(pred)
                            
                    # compute losses
                    if cfg_da.use_domain_adaptation:
                        loss_kl_tot = loss_kl + loss_kl_t
                        loss_kl_tot *= cfg_da.da_info['kl_weight']

                        ind_max_cls = torch.argmax(gcr_pre_t.detach(), 1)
                        for i_state in ind_max_cls:
                            self.new_state[i_state] += 1
                        w1 = da_I3Net.dcbr_w1_weight(gcr_pre_t.sigmoid().detach())
                        w2 = torch.exp(1 - self.old_state[ind_max_cls]/img_per_epoch)
                        if epoch >= cfg_da.da_info['open_all_loss_epoch_idx'] and torch.sum(self.old_state) > 100:
                            weight = (w1+w2)*0.5 
                        else:
                            weight = torch.ones(w1.size(0)).to(w1.device)  

                        dloss_l = 0.5 * torch.mean(domain_l ** 2) * cfg_da.da_info['dloss_l_weight']
                        dloss_g = 0.5 * da_I3Net.weight_ce_loss(domain_g, 0, torch.ones(domain_g.size(0)).to(domain_g.device)) * 0.1

                        dloss_l_t = 0.5 * torch.mean((1-domain_l_t) ** 2) * cfg_da.da_info['dloss_l_weight']
                        dloss_g_t = 0.5 * da_I3Net.weight_ce_loss(domain_g_t, 1, weight) * cfg_da.da_info['dcbr_weight'] 
                        loss_gf = 38 * torch.pow(g_feat-g_feat_t, 2.0).mean()

                        cls_onehot = da_I3Net.gt_classes2cls_onehot(len(gcr_pre), batch['cls'], batch['batch_idx'], self.model.nc)
                        cls_onehot = torch.from_numpy(cls_onehot).to(gcr_pre.device)
                        loss_gcr = nn.BCEWithLogitsLoss()(gcr_pre, cls_onehot) * cfg_da.da_info['gcr_weight']
                    # loss.sum() * batch_size, loss.detach()
                    self.loss, self.loss_items = self.criterion(preds, batch)

                    if cfg_ss.use_semisupervised_in_loop and batch_unlabelled is not None:
                        # images_ss = torch.cat(images_ss, dim=0).view(cfg_ss.batch_size, 3, cfg_ss.image_size, cfg_ss.image_size)
                        images_ss = torch.cat(images_ss, dim=0).view(-1, 3, cfg_ss.image_size, cfg_ss.image_size)
                        images_ss = images_ss.to(self.device)
                        images_teacher_dict = {}
                        images_teacher_dict['img'] = images_ss
                        images_teacher_dict = self.validator.preprocess(images_teacher_dict)
                        images_ss = images_teacher_dict['img'] 
                        targets_ss = targets_ss
                       
                        im_ss_h, im_ss_w = images_ss.shape[-2:]
                        assert im_ss_h==im_ss_w
                        if cfg_ss.use_strong_aug_for_student:
                            # add strong data aug here
                            # D:\BoyangDeng\Detection\ultralytics_yolov8\ultralytics\yolo\data\augment.py
                            # ultralytics.yolo.data.augment.Mosaic
                            # ultralytics.yolo.data.augment.MixUp
                            # images_ss_strong, targets_ss_strong = self.ss_td_loader.dataset.transforms_strong(images_ss, targets_ss)
                            from ..utils.instance import Instances
                            label_ss_list = []
                            for i_im in range(len(images_ss)):
                                one_label = {}
                                im_ss = images_ss[i_im]
                                im_ss = im_ss.permute(1, 2, 0)
                                target_ss = targets_ss[i_im]
                                one_im_gt = target_ss['labels'].unsqueeze(1).cpu().numpy()
                                one_im_bboxes = target_ss['boxes'].cpu().numpy()
                                assert im_ss_h == im_ss_w
                                one_im_bboxes = np.clip(one_im_bboxes, 0, im_ss_w)
                                if len(one_im_gt) == 0:
                                    break
                                one_label['im_file'] = ''
                                one_label['img'] = im_ss
                                one_label['segments'] = []
                                one_label['keypoints'] = None
                                one_label['normalized'] = True
                                one_label['bbox_format'] = 'xyxy'
                                one_label['ratio_pad'] = (1.0, 1.0)
                                one_label['cls'] = one_im_gt
                                # one_label['bboxes'] = one_im_bboxes
                                
                                new_instances = Instances(one_im_bboxes, segments=[], keypoints=None, bbox_format=one_label['bbox_format'], normalized=False)
                                one_label['instances'] = new_instances
                                one_label['ori_shape'] = (im_ss_h, im_ss_w)
                                one_label['resized_shape'] = (im_ss_h, im_ss_w)
                                label_ss_list.append(one_label)

                            invalid_label_idxes = []
                            for i_label, one_label in enumerate(label_ss_list):
                                tmp_im = one_label['img'].cpu().numpy()
                                need_rescale = False
                                if tmp_im.max() <= 1:
                                    import cv2
                                    need_rescale = True
                                    tmp_im = cv2.normalize(tmp_im.astype(np.float32), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                                    tmp_im = tmp_im.astype(np.uint8)
                                try:
                                    tmp_label = cfg_ss.deep_copy_one_label(one_label)
                                    tmp_label['img'] = tmp_im
                                    tmp_label = model_albumentations(tmp_label)
                                    tmp_label = model_randomHSV(tmp_label)
                                    if need_rescale:
                                        tmp_label['img'] = cv2.normalize(tmp_label['img'], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                    tmp_label['img'] = torch.tensor(tmp_label['img'])
                                    one_label = tmp_label
                                except Exception as e:
                                    print(e)
                                    print(traceback.format_exc())
                                    invalid_label_idxes.append(i_label)
                            if len(invalid_label_idxes) > 0:
                                label_ss_list_new = []
                                for i_label, one_label in enumerate(label_ss_list):
                                    if i_label not in invalid_label_idxes:
                                        label_ss_list_new.append(one_label)
                                label_ss_list = label_ss_list_new
                                # cfg_ss.show_image_with_bbox(one_label['img'],  one_label['instances'].bboxes, show=True)
                            label_num = len(label_ss_list)
                            
                            if cfg_ss.use_mosaic_and_mixup_for_student and label_num>=2:
                                label_ss_list_new = []
                                labels = label_ss_list[0]
                                label_ss_list_new.append(labels)
                                labels_mosaic = cfg_ss.deep_copy_one_label(labels)
                                labels_mosaic['mix_labels'] = []
                                if label_num >= 4:
                                    for tmp_label in label_ss_list[1:4]:
                                        mix_label = cfg_ss.deep_copy_one_label(tmp_label)
                                        labels_mosaic['mix_labels'].append(mix_label)
                                else:
                                    tmp_list= np.random.choice(label_ss_list, 3, replace=True)
                                    for tmp_label in tmp_list:
                                        mix_label = cfg_ss.deep_copy_one_label(tmp_label)
                                        labels_mosaic['mix_labels'].append(mix_label)
                                labels1 = model_mosaic._mix_transform(labels_mosaic, use_tensor=True)
                                # if cfg_tot.is_debug:
                                #     cfg_ss.show_image_with_bbox(labels1['img'],  labels1['instances'].bboxes, show=True)
                                import torch.nn.functional as F
                                im_tmp = labels1['img'].permute(2, 0, 1).unsqueeze(0)
                                # scale = float(im_ss_w) / labels1['resized_shape'][0]
                                scale=  0.5
                                im_tmp = F.interpolate(im_tmp, size=(im_ss_w, im_ss_h), mode='bilinear').to(self.device)
                                labels1['img'] = im_tmp.squeeze(0).permute(1, 2, 0)
                                labels1['instances'].scale(scale_w=scale, scale_h=scale, bbox_only=True)
                                labels1['resized_shape'] = (im_ss_h, im_ss_w)
                                label_ss_list_new.append(labels1)
                                if label_num >= 4:
                                    labels_mixup = cfg_ss.deep_copy_one_label(labels)
                                    # tmp_labels = np.random.choice(label_ss_list[1:], 1, replace=True)[0]
                                    tmp_labels = label_ss_list[1]
                                    mix_labels = cfg_ss.deep_copy_one_label(tmp_labels)
                                    labels_mixup['mix_labels'] = [mix_labels]
                                    labels2 = model_mixup._mix_transform(labels_mixup, use_tensor=True)
                                    # if cfg_tot.is_debug:
                                    #     cfg_ss.show_image_with_bbox(labels2['img'],  labels2['instances'].bboxes, show=True)
                                    labels_mixup = cfg_ss.deep_copy_one_label(label_ss_list[2])
                                    # tmp_labels = np.random.choice(label_ss_list[1:], 1, replace=True)[0]
                                    tmp_labels = label_ss_list[3]
                                    mix_labels = cfg_ss.deep_copy_one_label(tmp_labels)
                                    labels_mixup['mix_labels'] = [mix_labels]
                                    labels3 = model_mixup._mix_transform(labels_mixup, use_tensor=True)
                                    # if cfg_tot.is_debug:
                                    #     cfg_ss.show_image_with_bbox(labels3['img'],  labels3['instances'].bboxes, show=True)
                                    label_ss_list_new.append(labels2)
                                    label_ss_list_new.append(labels3)
                                elif label_num >= 2:
                                    labels_mixup = cfg_ss.deep_copy_one_label(labels)
                                    # tmp_labels = np.random.choice(label_ss_list[1:], 1, replace=True)[0]
                                    tmp_labels = label_ss_list[1]
                                    mix_labels = cfg_ss.deep_copy_one_label(tmp_labels)
                                    labels_mixup['mix_labels'] = [mix_labels]
                                    labels2 = model_mixup._mix_transform(labels_mixup, use_tensor=True)
                                    label_ss_list_new.append(labels2)
                                # else:
                                #     labels2 = labels
                                #     labels3 = labels
                                label_ss_list = label_ss_list_new
                            # (n ,3, 960, 960)      
                            targets_ss = []
                            images_ss = None
                            for one_label in label_ss_list:
                                im = one_label['img'].permute(2, 0, 1).unsqueeze(0).to(self.device)
                                if images_ss is None:
                                    images_ss = im
                                else:
                                    images_ss = torch.concat([images_ss, im], 0)
                                target_ss = {}
                                target_ss['labels'] = torch.tensor(one_label['cls'])
                                target_ss['boxes'] = torch.tensor(one_label['instances'].bboxes)
                                targets_ss.append(target_ss)

                        for i_im in range(len(label_ss_list)):
                            target_ss = targets_ss[i_im]
                            if len(target_ss['labels'].shape) == 1:
                                one_im_gt = target_ss['labels'].unsqueeze(1)
                            else:
                                one_im_gt = target_ss['labels']
                            one_im_bboxes = target_ss['boxes']

                            if cfg_ss.save_teacher_predicted_im and i % cfg_ss.save_teacher_predicted_im_interval == 0:
                                cur_bboxes = target_ss['boxes'].cpu().numpy()
                                cur_label = target_ss['labels'].cpu().numpy()
                                cur_gt = np.concatenate([cur_bboxes, cur_label], 1)
                                prefix = 'epoch_'+str(epoch)+'_step_'+str(i)+'_'
                                cfg_ss.show_image_with_bbox(images_ss[i_im], cur_gt, save_dir=self.save_dir, prefix=prefix)

                            one_im_bboxes = cfg_ss.xyx1y1_to_cxcywh(one_im_bboxes, im_ss_w, im_ss_h)
                            if i_im == 0:
                                label_ss = {}
                                label_ss['im_file'] = ['']
                                label_ss['cls'] = one_im_gt
                                label_ss['bboxes'] = one_im_bboxes
                                label_ss['ori_shape'] = [[im_ss_h, im_ss_w]]
                                label_ss['resized_shape'] = [[im_ss_h, im_ss_w]]
                                label_ss['batch_idx'] = torch.tensor([i_im]*len(label_ss['cls']))
                            else:
                                label_ss['im_file'] += ['']
                                label_ss['cls'] = torch.concat([label_ss['cls'], one_im_gt], 0)
                                label_ss['bboxes'] = torch.concat([label_ss['bboxes'], one_im_bboxes], 0)
                                label_ss['ori_shape'] += [[im_ss_h, im_ss_w]]
                                label_ss['resized_shape'] += [[im_ss_h, im_ss_w]]
                                label_ss['batch_idx'] = torch.concat([label_ss['batch_idx'], torch.tensor([i_im]*len(one_im_gt))], 0)
                        loss_ss = 0
                        if images_ss is not None:
                            label_ss['img']  = images_ss
                            label_ss['bboxes'] = label_ss['bboxes'].to(self.device)
                            label_ss['batch_idx'] = label_ss['batch_idx'].to(self.device)
                            label_ss['cls'] = label_ss['cls'].to(self.device)
                            # cfg_ss.show_image_with_bbox(labels3['img'],  labels3['instances'].bboxes, show=True)
                            images_ss_formatted = images_ss
                            targets_ss_formatted = label_ss
                            images_ss_formatted = images_ss_formatted.to(torch.float32)
                            preds_ss = self.model(images_ss_formatted)
                            # if cfg_tot.is_debug:
                            #     pred_final = self.validator.postprocess(preds_ss, conf=cfg_ss.teacher_lowest_postprocess_conf_t, iou=0.5)[0]
                            loss_ss, loss_items_ss = self.criterion(preds_ss, targets_ss_formatted)
                            # loss_ss = loss_ss * 0.1
                            loss_ss = loss_ss * 0.5

                    # add losses
                    if cfg_da.use_domain_adaptation:
                        loss_da = torch.zeros(8, device=self.device)
                        loss_da[0] = dloss_l
                        loss_da[1] = dloss_l_t
                        loss_da[2] = dloss_g
                        loss_da[3] = dloss_g_t
                        loss_da[4] = loss_gf
                        loss_da[5] = loss_gcr
                        if epoch >= cfg_da.da_info['open_all_loss_epoch_idx'] and torch.sum(self.old_state) > 100:
                            loss_gpa = da_I3Net.get_pa_losses(fea_lists, fea_lists_t) * cfg_da.da_info['pa_losses_weight']
                            loss_da[6] = loss_gpa
                            loss_da[7] = loss_kl_tot
                        self.loss += loss_da.sum()
                        self.loss_items = torch.cat((self.loss_items, loss_da.detach()))

                    if cfg_ss.use_semisupervised_in_loop and batch_unlabelled is not None:
                        if loss_ss > 0:
                            if loss_items_ss[0].cpu().numpy()[()] > 0:
                                self.loss += loss_ss.sum()
                                self.loss_items = torch.cat((self.loss_items, loss_items_ss))

                    if RANK != -1:
                        self.loss *= world_size
                    try:
                        if self.tloss is not None:
                            loss_num = min(len(self.loss_items), len(self.tloss))
                            self.tloss = (self.tloss[:loss_num] * i + self.loss_items[:loss_num]) / (i + 1)  
                        else:
                            self.tloss = self.loss_items
                        if cfg_ss.use_semisupervised_in_loop:
                            if len(self.loss_items) > len(self.tloss):
                                self.tloss = self.loss_items
                    except Exception as e:
                        print(e)

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    if cfg_ss.use_semisupervised_in_loop and cfg_ss.use_semi_ema_as_teacher:
                        if ni >= nw: # after warmup
                            if not cfg_ss.use_ema_as_teacher:
                                self.ss_framework.semi_ema.update(self.ema.ema)

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

                if cfg_da.use_domain_adaptation:
                    self.old_state = self.new_state
                    #print(self.old_state.cpu().numpy())

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):
                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                try:
                    if cfg_tot.is_debug:
                        if epoch > 0 and epoch % cfg_tot.val_epoch_interval == 0:
                            if self.args.val or final_epoch:
                                self.metrics, self.fitness = self.validate()
                            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                    else:
                        if self.args.val or final_epoch:
                            self.metrics, self.fitness = self.validate()
                        self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                except Exception as e:
                    print(e)
                    import traceback
                    print(traceback.format_exc())
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')

            if cfg_ss.use_semisupervised_in_loop:
                self.ss_framework.on_train_epoch_end(self.ss_td_loader, epoch)

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        if cfg_da.use_domain_adaptation:
            ckpt = {
                'epoch': self.epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(de_parallel(self.model)).half(),
                'model_L2Norm': deepcopy(de_parallel(self.L2Norm)).half(),
                'model_netD_pixel': deepcopy(de_parallel(self.netD_pixel)).half(),
                'model_netD': deepcopy(de_parallel(self.netD)).half(),
                'model_conv_gcr': deepcopy(de_parallel(self.conv_gcr)).half(),
                'model_RandomLayer': deepcopy(de_parallel(self.RandomLayer)).half(),
                'old_state': deepcopy(de_parallel(self.old_state)).half(),
                'new_state': deepcopy(de_parallel(self.new_state)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'train_args': vars(self.args),  # save as dict
                'date': datetime.now().isoformat(),
                'version': __version__}
        elif cfg_ss.use_semisupervised and self.epoch >= cfg_ss.start_epoch:
            ckpt = {
                'epoch': self.epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'semi_ema': deepcopy(self.ss_framework.semi_ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'train_args': vars(self.args),  # save as dict
                'date': datetime.now().isoformat(),
                'version': __version__}
        else:
            ckpt = {
                'epoch': self.epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'train_args': vars(self.args),  # save as dict
                'date': datetime.now().isoformat(),
                'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt

    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized.
        """
        return data['train'], data.get('val') or data.get('test')

    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        if cfg_da.use_domain_adaptation:
            torch.nn.utils.clip_grad_norm_(self.L2Norm.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.netD_pixel.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.conv_gcr.parameters(), max_norm=10.0)  # clip gradients
            torch.nn.utils.clip_grad_norm_(self.RandomLayer.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        try:
            metrics = self.validator(self)
            fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
            if not self.best_fitness or self.best_fitness < fitness:
                self.best_fitness = fitness
        except Exception as e:
            print(e)
            fitness = 0
            import traceback
            print(traceback.format_exc())
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        raise NotImplementedError('get_validator function not implemented in trainer')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train', domain_adaptation=False):
        """
        Returns dataloader derived from torch.data.Dataloader.
        """
        raise NotImplementedError('get_dataloader function not implemented in trainer')

    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor.
        """
        raise NotImplementedError('criterion function not implemented in trainer')

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        self.model.names = self.data['names']

    def build_targets(self, preds, targets):
        pass

    def progress_string(self, loss_names):
        return ''

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        pass

    def final_eval(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def check_resume(self):
        resume = self.args.resume
        if resume:
            try:
                last = Path(
                    check_file(resume) if isinstance(resume, (str,
                                                              Path)) and Path(resume).exists() else get_latest_run())
                self.args = get_cfg(attempt_load_weights(last).args)
                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume

    def resume_training(self, ckpt):
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without --resume, i.e. 'yolo task=... mode=train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)

    @staticmethod
    def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        g = [], [], []  # optimizer parameter groups
        ad = [], [], []  # optimizer parameter groups
        adnet_gcr = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if cfg_da.use_domain_adaptation:
            for i_module in range(len(model)):
                if i_module == 0:
                    for v in model[i_module].modules():
                        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                            g[2].append(v.bias)
                        if isinstance(v, bn):  # weight (no decay)
                            g[1].append(v.weight)
                        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                            g[0].append(v.weight)
                elif i_module == -1:
                    for v in model[i_module].modules():
                        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                            adnet_gcr[2].append(v.bias)
                        if isinstance(v, bn):  # weight (no decay)
                            adnet_gcr[1].append(v.weight)
                        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                            adnet_gcr[0].append(v.weight)
                else:
                    for v in model[i_module].modules():
                        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                            ad[2].append(v.bias)
                        if isinstance(v, bn):  # weight (no decay)
                            ad[1].append(v.weight)
                        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                            ad[0].append(v.weight)
        else:
            for v in model.modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                    g[2].append(v.bias)
                if isinstance(v, bn):  # weight (no decay)
                    g[1].append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    g[0].append(v.weight)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')
        if cfg_da.use_domain_adaptation:
            optimizer.add_param_group({'params': g[0], 'weight_decay': decay, 'lr': lr, 'momentum': momentum})  # add g0 with weight_decay
            optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0, 'lr': lr, 'momentum': momentum})  # add g1 (BatchNorm2d weights)

            optimizer.add_param_group({'params': ad[2], 'weight_decay': 0.0, 'lr': cfg_da.da_info['lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': ad[0], 'weight_decay': decay, 'lr': cfg_da.da_info['lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': ad[1], 'weight_decay': 0.0, 'lr': cfg_da.da_info['lr'], 'momentum': momentum, 'nesterov': True})

            optimizer.add_param_group({'params': adnet_gcr[2], 'weight_decay': 0.0, 'lr': cfg_da.da_info['net_gcr_lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': adnet_gcr[0], 'weight_decay': decay, 'lr': cfg_da.da_info['net_gcr_lr'], 'momentum': momentum, 'nesterov': True})
            optimizer.add_param_group({'params': adnet_gcr[1], 'weight_decay': 0.0, 'lr': cfg_da.da_info['net_gcr_lr'], 'momentum': momentum, 'nesterov': True})
        else:
            optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
            optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                    f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
        return optimizer


def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model.
    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Returns:
        bool: Returns True if the AMP functionality works correctly with YOLOv8 model, else False.

    Raises:
        AssertionError: If the AMP checks fail, indicating anomalies with the AMP functionality on the system.
    """
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices

    def amp_allclose(m, im):
        # All close FP32 vs AMP results
        a = m(im, device=device, verbose=False)[0].boxes.boxes  # FP32 inference
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.boxes  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    f = ROOT / 'assets/bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if ONLINE else np.ones((640, 640, 3))
    prefix = colorstr('AMP: ')
    LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...')
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO('yolov8n.pt'), im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. Setting 'amp=True'.")
    except AssertionError:
        LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                       f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
        return False
    return True
