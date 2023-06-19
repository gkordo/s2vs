import os
import random
import utils
import h5py
import glob
import torch
import numpy as np

from abc import abstractmethod
from torch.utils.data import Dataset


class HDF5DatasetGenerator(Dataset):
    def __init__(self, feature_file, videos, min_len=4, dims=512):
        super(HDF5DatasetGenerator, self).__init__()
        self.feature_file = h5py.File(feature_file, "r")
        self.videos = videos
        self.min_len = min_len
        self.dims = dims

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            video_id = self.videos[idx]
            features = self.feature_file[video_id][:]
            while features.shape[0] < self.min_len:
                features = np.concatenate([features, features], axis=0)
            if features.ndim == 2:
                features = np.expand_dims(features, 1)
            features = torch.from_numpy(features.astype(np.float32))
            return features, video_id
        except Exception as e:
            return torch.zeros((0, 1, self.dims)), ''


class TrainDataset(Dataset):

    def __init__(self, dataset_path, weak_aug=None, strong_aug=None, window_sz=32):
        super(TrainDataset, self).__init__()
        self.dataset_path = dataset_path
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.window_sz = window_sz

    @abstractmethod
    def next_epoch(self):
        pass

    def collate_fn(self, batch):
        weak_videos, strong_videos = zip(*batch)
        weak_videos = torch.stack(weak_videos)
        strong_videos = torch.stack(strong_videos)
        labels = torch.eye(weak_videos.shape[0])

        strong_videos, labels_viv = self.strong_aug.mixup(strong_videos, torch.eye(strong_videos.shape[0]))
        labels = torch.cat([labels, labels_viv])

        videos = torch.cat([weak_videos, strong_videos])
        labels = labels.matmul(labels.transpose(0, 1)).bool().float()

        return videos, labels


class SSLGenerator(TrainDataset):

    def __init__(self, dataset_path, weak_aug=None, strong_aug=None, window_sz=32, percentage=1., **kargs):
        super(SSLGenerator, self).__init__(dataset_path, weak_aug, strong_aug, window_sz)
        self.dataset_path = dataset_path.split(',')
        self.videos = []
        for vdir in self.dataset_path:
            self.videos += [v.replace('/00000.jpg', '') for v in glob.glob(
                os.path.join(vdir, '*/00000.jpg'))]
        self.videos = sorted(self.videos)
        self.videos = self.videos[:int(len(self.videos)*percentage)]

    def next_epoch(self):
        random.shuffle(self.videos)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_dir = self.videos[idx]

        video = utils.load_frames(video_dir=video_dir, window=self.window_sz*2)

        weak_video = self.weak_aug(video.copy())
        strong_video = self.strong_aug(video.copy())

        return weak_video, strong_video


class VideoDatasetGenerator(Dataset):
    def __init__(self, dataset_path, videos, pattern, loader='video', fps=1, crop=None, resize=None):
        super(VideoDatasetGenerator, self).__init__()
        self.dataset_path = dataset_path
        self.videos = videos
        self.pattern = pattern
        self.loader = loader
        self.fps = fps
        self.crop = crop
        self.resize = resize

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            if self.loader == 'video':
                video = glob.glob(os.path.join(self.dataset_path, self.pattern.replace('{id}', self.videos[idx])))
                video = utils.load_video_ffmpeg(video[0], fps=self.fps, crop=self.crop, resize=self.resize)
                # video = load_video_opencv(video[0], fps=self.fps, crop=self.crop, resize=self.resize)
            elif self.loader == 'frame':
                frame_dir = os.path.join(self.dataset_path, self.pattern.replace('{id}', self.videos[idx]))
                video = utils.load_frames_opencv(frame_dir, crop=self.crop, resize=self.resize)
            return torch.from_numpy(video.copy()), self.videos[idx]
        except:
            return torch.tensor([]), ''
