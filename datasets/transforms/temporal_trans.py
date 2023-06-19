import numpy as np
import torch.nn as nn


class TemporalShuffleDropout(nn.Module):

    def __init__(self,
                 window_range=(4, 16),
                 p_shuffle=0.5,
                 p_dropout=0.3,
                 p_content=0.5,
                 ):
        super().__init__()
        self.window_range = window_range
        self.p_shuffle = p_shuffle
        self.p_dropout = p_dropout
        self.p_content = p_content

    def _reshape_video_ids(self, idx):
        T = idx.shape[0]
        window_size = np.random.randint(*self.window_range)
        offset = T % window_size
        if T > offset:
            idx = idx[:T - offset]
        else:
            window_size = T
        return idx.reshape(-1, window_size)

    def _shuffle(self, idx):
        if np.random.uniform() < self.p_shuffle:
            np.random.shuffle(idx)
        return idx

    def _get_dropout_mask(self, idx):
        dropout_mask = np.random.rand(idx.shape[0]) < self.p_dropout
        dropout_mask = dropout_mask.astype(int)
        dropout_mask *= 1 + (np.random.rand(idx.shape[0]) > self.p_content).astype(int)
        return dropout_mask

    def _dropout(self, idx, seg_id):
        np.delete(idx, seg_id, 0)
        return idx

    def _drop_content(self, video, idx):
        _, H, W, C = video.shape
        for i in idx:
            if np.random.uniform() > 0.5:
                video[i] = 0.
            else:
                video[i] = np.random.rand(1, H, W, C) * 255
        return video

    def forward(self, video):
        idx = np.arange(len(video))
        idx = self._reshape_video_ids(idx)
        idx = self._shuffle(idx)

        dropout_mask = self._get_dropout_mask(idx)
        if 0 < np.sum(dropout_mask) < idx.shape[0]:
            content_mask = idx[dropout_mask == 2].reshape(-1)
            video = self._drop_content(video, content_mask)
            idx = idx[~(dropout_mask == 1)]

        idx = idx.reshape(-1)
        return video[idx]


class FastForward(nn.Module):

    def __init__(self, ff_range):
        super().__init__()
        self.ff_range = ff_range

    def forward(self, video):
        if isinstance(self.ff_range, tuple):
            fast = np.random.randint(*self.ff_range)
        else:
            fast = self.ff_range
        return video[::fast]


class SlowMotion(nn.Module):

    def __init__(self, sm_range):
        super().__init__()
        self.sm_range = sm_range

    def forward(self, video):
        if isinstance(self.sm_range, tuple):
            slow = np.random.randint(*self.sm_range)
        else:
            slow = self.sm_range
        idx = np.repeat(np.arange(len(video)), slow)
        return video[idx]


class ReverseVideo(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, video):
        return video[::-1]
    
    
class PauseVideo(nn.Module):

    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len

    def forward(self, video):
        idx = np.arange(len(video))
        start = np.random.randint(len(idx) - 1)
        dur = np.random.randint(self.max_len)
        segment = np.repeat(idx[start:start + 1], dur)
        idx = np.insert(idx, start, segment, axis=0)
        return video[idx]


class TemporalTransformations(nn.Module):

    def __init__(
            self,
            window_sz,
            p_tsd=0.5,
            p_ff=0.1,
            p_sm=0.1,
            p_rev=0.1,
            p_pau=0.1,
            tsd_window_range=(4, 16),
            p_shuffle=0.5,
            p_dropout=0.3,
            p_content=0.5,
            ff_range=(2, 4),
            sm_range=2,
            **kargs
    ):
        super().__init__()
        self.window_sz = window_sz

        self.p = [
            p_tsd,
            p_ff,
            p_sm,
            p_rev,
            p_pau
        ]
        self.p.append(np.maximum(0., 1. - np.sum(self.p)))

        self.tsd = TemporalShuffleDropout(
            tsd_window_range,
            p_shuffle,
            p_dropout,
            p_content)
        self.ff = FastForward(ff_range)
        self.sm = SlowMotion(sm_range)
        self.rv = ReverseVideo()
        self.pv = PauseVideo(window_sz // 4)

    def forward(self, video):
        trn = np.random.choice(len(self.p), p=self.p)
        if trn == 0:
            video = self.tsd(video)
        elif trn == 1:
            video = self.ff(video)
        elif trn == 2:
            video = self.sm(video)
        elif trn == 3:
            video = self.rv(video)
        elif trn == 4:
            video = self.pv(video)
        return video

    def __repr__(self):
        return '{}(p_tsd={}, p_ff={}, p_sm={}, p_rev={}, p_pau={})'.format(self.__class__.__name__, *self.p)
