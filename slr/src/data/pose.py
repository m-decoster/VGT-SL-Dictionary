import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from . import custom_transforms as CT
from .common import Sample, collect_samples
from scipy.ndimage import gaussian_filter1d


def get_augmentation_transforms(job):
    """Get the transforms required to perform augmentation. For validation and testing, these do nothing.

    :param job: The job."""
    if not job == 'train':
        return CT.Passthrough()
    return T.Compose([
        CT.ShiftHandFrames(p=0.2),
        CT.RandomHorizontalFlip(p=0.2),
        CT.RotateHandsIndividually(p=0.2),
        CT.Jitter(p=0.2),
        CT.RandomCrop(p=0.2),
        CT.DropFrames(p=0.2, drop_ratio=0.1),
        CT.FrameHandDropout(p=0.2, drop_ratio=0.1)
    ])


class PoseDataset(Dataset):
    """Used during training and testing: loads keypoints."""

    def __init__(self, job: str, root_path: str, retrain_on_all: bool, cut_transients: bool):
        """Create a PoseDataset for training or testing.

        :param job: "train", "val" or "test".
        :param root_path: The root data directory.
        :param retrain_on_all: If provided, will load the validation set as part of the training set.
        """
        super(PoseDataset, self).__init__()

        self.root_path = root_path
        self.job = job
        self.retrain_on_all = retrain_on_all
        self.cut_transients = cut_transients

        self.augment = get_augmentation_transforms(self.job)

        self.samples = self._collect_samples()

    def __getitem__(self, item):
        sample = self.samples[item]
        sample_filename, _ = os.path.splitext(sample.path)
        clip = np.load(os.path.join(self.root_path, 'mediapipe', f'{sample_filename}.npy'))

        clip = torch.from_numpy(clip).float()

        if self.cut_transients:
            clip = self._remove_transients(clip)

        clip = self.augment(clip)

        return clip, sample.label, sample.path

    def __len__(self):
        return len(self.samples)

    def _collect_samples(self) -> List[Sample]:
        return collect_samples(self.root_path, self.job, self.retrain_on_all)

    def _remove_transients(self, clip):
        rp = clip[:, 13, :]
        lp = clip[:, 14, :]
        vr = np.diff(-rp[..., 1], prepend=-rp[0, 1])
        vr = gaussian_filter1d(vr, 2)
        vl = np.diff(-lp[..., 1], prepend=-lp[0, 1])
        vl = gaussian_filter1d(vl, 2)
        ar = np.diff(vr, prepend=vr[0])
        al = np.diff(vl, prepend=vl[0])

        jr = np.diff(ar, prepend=ar[0])
        jl = np.diff(al, prepend=al[0])

        cross_r = np.where(np.diff(np.sign(jr)))[0]
        cross_l = np.where(np.diff(np.sign(jl)))[0]
        cutoff_start, cutoff_end = min(cross_r[3], cross_l[3]), max(cross_r[-2], cross_l[-2])

        if len(clip[cutoff_start:cutoff_end]) > 0:
            return clip[cutoff_start:cutoff_end]
        else:
            return clip
