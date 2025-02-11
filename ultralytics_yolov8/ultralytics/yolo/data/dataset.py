# Ultralytics YOLO 🚀, GPL-3.0 license

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import os
import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from ..utils import LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable
from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label
import config as cfg
from typing import Any, Dict, Generator, List, Optional, Tuple
from collections import defaultdict
import config_ss as cfg_ss

class YOLODataset(BaseDataset):
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    """
    Dataset class for loading images object detection and/or segmentation labels in YOLO format.

    Args:
        img_path (str): path to the folder containing images.
        imgsz (int): image size (default: 640).
        cache (bool): if True, a cache file of the labels is created to speed up future creation of dataset instances
        (default: False).
        augment (bool): if True, data augmentation is applied (default: True).
        hyp (dict): hyperparameters to apply data augmentation (default: None).
        prefix (str): prefix to print in log messages (default: '').
        rect (bool): if True, rectangular training is used (default: False).
        batch_size (int): size of batches (default: None).
        stride (int): stride (default: 32).
        pad (float): padding (default: 0.0).
        single_cls (bool): if True, single class training is used (default: False).
        use_segments (bool): if True, segmentation masks are used as labels (default: False).
        use_keypoints (bool): if True, keypoints are used as labels (default: False).
        names (list): class names (default: None).

    Returns:
        A PyTorch dataset object that can be used for training an object detection or segmentation model.
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=None,
                 prefix='',
                 rect=False,
                 batch_size=None,
                 stride=32,
                 pad=0.0,
                 single_cls=False,
                 use_segments=False,
                 use_keypoints=False,
                 names=None,
                 classes=None,
                 need_label=True):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.names = names

        if cfg_ss.use_semisupervised:
            self.labelled_dataset = None
            self._distribution_per_class: Dict[int, np.ndarray] = {}
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(img_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad, single_cls, classes, need_label=need_label)

    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.names))))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            im_idx = 0
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                im_idx += 1
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                else:
                    print('imagee is not valid:', im_idx)
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
        return x

    def get_labels(self, need_label=True):
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        if cfg.merge_labels:
            for one_im_info in labels:
                for i_cls in range(len(one_im_info['cls'])):
                    cls_idx = one_im_info['cls'][i_cls][0]
                    if cls_idx in cfg.merge_labels_dict:
                        one_im_info['cls'][i_cls][0] = cfg.merge_labels_dict[cls_idx]

        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if need_label:
            if len_cls == 0:
                raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here"""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        keys_common = []
        values = []
        include_num = 0
        for key in keys:
            include_num = 0
            for b in batch:
                if key in b:
                    include_num += 1
            if include_num == len(batch):
                keys_common.append(key)
                value_one_key = []
                for b in batch:
                    value = b[key]
                    value_one_key.append(value)
                values.append(value_one_key)
        # values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys_common):
            try:
                value = values[i]
                if k == 'img':
                    value = torch.stack(value, 0)
                if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                    value = torch.cat(value, 0)
            except Exception as e:
                # print(e)
                # import traceback
                # print(traceback.format_exc())
                continue
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch

    # use_semisupervised = True
    def build_transform_for_SSOD(self, hyp):
        # hyp.scale = 0
        # hyp.fliplr = 0
        # hyp.flipud = 0
        hyp.mosaic = 1.0
        hyp.mixup = 1.0
        hyp.copy_paste = 0.0  
        self.transforms_strong = self.build_transforms(hyp=hyp)

        hyp.degrees = 0
        hyp.translate = 0
        hyp.shear = 0
        hyp.perspective = 0
        hyp.hsv_h = 0
        hyp.hsv_s = 0
        hyp.hsv_v = 0

        hyp.fliplr = 0.5

        hyp.mosaic = 0.0
        hyp.mixup = 0.0  
        hyp.copy_paste = 0.0  
        self.transforms_weak = self.build_transforms(hyp)
        self.transforms = self.transforms_weak

    # def random_unlabelled_sample(self,) -> Generator[torch.Tensor, None, None]:
    def random_unlabelled_sample(self,):
        from random import choice, sample, uniform

        if cfg.is_debug:
            indices_to_sample = sample(list(range(self.__len__())), k=cfg_ss.random_unlabelled_sample_num)
            im_list = []
            # for idx in tqdm(indices_to_sample):
            #     preprocessed_im = self.__getitem__(idx)
            #     im_list.append(preprocessed_im)
            # return im_list
            print('indices_to_sample[::10]:', indices_to_sample[:10])
            for idx in indices_to_sample:
                yield self.__getitem__(idx)
        else:
            """
            no yield, otherwise the return will be a generator even within the "if cfg.is_debug:"
            assume the unlabeled set always larger than the labeled set
            """
            unlabled_im_num = self.__len__()
            labled_im_num = len(self.labelled_dataset)
            sample_num = min(unlabled_im_num, labled_im_num)
            if cfg_ss.limit_sample_num > 0:
                sample_num = min(sample_num, cfg_ss.limit_sample_num)
            indices_to_sample = sample(list(range(unlabled_im_num)), k=sample_num)
            # for idx in tqdm(indices_to_sample):
            #     preprocessed_im = self.__getitem__(idx)
            #     im_list.append(preprocessed_im)
            #     return im_list
            print('indices_to_sample[::10]:', indices_to_sample[:10])
            for idx in indices_to_sample:
                yield self.__getitem__(idx)

    def _calculate_class_distributions(self) -> None:
        print("Calculating real distributions per class")
        # we set the max to 1 because 0 is the background class,
        # so we will ignore it

        # max_label_idx = -1
        max_label_idx = len(cfg_ss.CLASSNAME_TO_IDX) - 1
        num_per_class = defaultdict(list)
        cnt_t = cfg_ss.sample_num_for_calculate_class_distributions

        for idx in tqdm(range(len(self.labelled_dataset))):
            if cnt_t > 0 and idx > cnt_t:
                break
            # _, targets = self.labelled_dataset[idx]
            targets = self.labelled_dataset[idx]
            labels = targets["cls"].detach().cpu().numpy().astype(int).squeeze().tolist()
            if not isinstance(labels, list):
                labels = [labels]
            if len(labels) == 0:
                continue
            if max(labels) > max_label_idx:
                # fill in all the missing values with 0s
                max_label_idx = max(labels)
            #     num_filled_values = len(num_per_class[0])
            #     for label_idx in range(0, max_label_idx + 1):
            #         if len(num_per_class[label_idx]) < num_filled_values:
            #             missing = num_filled_values - len(num_per_class[label_idx])
            #             num_per_class[label_idx].extend([0] * missing)

            for label_idx in range(0, max_label_idx + 1):
                instance_num = sum(np.array(labels) == label_idx)
                # if instance_num>0:
                num_per_class[label_idx].append(instance_num)
        """
        the original small teacher model use idx 0 to indicate background
        """
        # key: np.array(val) for key, val in num_per_class.items() if key != 0
        self._distribution_per_class = {}
        for key, val in num_per_class.items():
            self._distribution_per_class[key] = np.array(val) 
        pass

    def check_class_distribution(self):
        if len(self._distribution_per_class) == 0:
            if cfg.is_debug:
                suffix = '_distribution_per_class_debug.npy'
            else:
                suffix = '_distribution_per_class.npy'
            save_path = os.path.join(cfg.data_dir, cfg.data_train+suffix)
            if os.path.exists(save_path):
                tmp = np.load(save_path, allow_pickle=True)
                self._distribution_per_class = {}
                tmp_dict = tmp[()]
                for key in tmp_dict:
                    self._distribution_per_class[key] = tmp_dict[key]
            else:
                self._calculate_class_distributions()
            np.save(save_path, self._distribution_per_class)
            tot_num = 0
            for class_idx in self._distribution_per_class:
                tot_num += sum(self._distribution_per_class[class_idx])
            print('_calculate_class_distributions tot instance num:', tot_num)
            for class_idx in self._distribution_per_class:
                sku_inst_num = sum(self._distribution_per_class[class_idx])
                if sku_inst_num > 0:
                    print(class_idx, sku_inst_num, round(sku_inst_num/tot_num, 2))

    def distribution_for_class(self, class_idx: int) -> torch.Tensor:
        self.check_class_distribution()
        return self._distribution_per_class[class_idx]

    @property
    def classes(self) -> List[int]:
        self.check_class_distribution()
        return list(self._distribution_per_class.keys())
    
# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.
    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        return len(self.samples)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):

    def __init__(self):
        pass
