import os
import sys
import cv2
import glob
import torch
import ffmpeg
import argparse
import numpy as np
import io as BytesIO
import seaborn as sns
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt

try:
    from IPython import display
    import imageio
except:
    pass


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier(device_ids=[int(args.rank)])
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    # This function disables printing when not in master process
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def collate_eval(batch):
    videos, video_ids = zip(*batch)
    num = len(videos)
    max_len = max([s.size(0) for s in videos])
    max_reg = max([s.size(1) for s in videos])
    dims = videos[0].size(2)

    padded_videos = videos[0].data.new(*(num, max_len, max_reg, dims)).fill_(0)
    masks = videos[0].data.new(*(num, max_len)).fill_(0)
    for i, tensor in enumerate(videos):
        length = tensor.size(0)
        padded_videos[i, :length] = tensor
        masks[i, :length] = 1

    return padded_videos, masks, video_ids


def batching(tensor, batch_sz):
    L = len(tensor)
    for i in range(L // batch_sz + 1):
        if i*batch_sz < L:
            yield tensor[i*batch_sz: (i+1)*batch_sz]


def save_model(args, model, optimizer, global_step, file_name='model.pth'):
    save_dict = {
        'args': args,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
    }
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    torch.save(save_dict, os.path.join(args.experiment_path, file_name))


def load_model(args, model, optimizer, file_name='model.pth'):
    print('>> loading network')
    d = torch.load(os.path.join(args.experiment_path, file_name), map_location='cpu')
    model.module.load_state_dict(d['model'])
    optimizer.load_state_dict(d['optimizer'])
    global_step = d.pop('global_step')
    return global_step

    
def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def pprint_args(args):
    print('\nInput Arguments')
    print('---------------')
    for k, v in sorted(dict(vars(args)).items()):
        print('%s: %s' % (k, str(v)))


def animate(frames, fps=1, save_file='./animation.gif'):
    if frames.dtype == np.float32:
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    imageio.mimsave(save_file, frames, fps=fps, loop=65535)
    with open(save_file,'rb') as f:
        display.display(display.Image(data=f.read()))


def random_crop(video, desired_size):
    H, W = video.shape[1:3]
    top = np.random.randint(np.maximum(1, (H - desired_size)/2))
    left = np.random.randint(np.maximum(1, (W - desired_size)/2))
    return video[:, top: top+desired_size, left: left+desired_size, :]


def center_crop(frame, desired_size):
    if frame.ndim == 3:
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[top: top+desired_size, left: left+desired_size, :]
    else:
        old_size = frame.shape[1:3]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[:, top: top+desired_size, left: left+desired_size, :]


def resize_frame(frame, desired_size):
    if isinstance(desired_size, int):
        min_size = np.min(frame.shape[:2])
        ratio = desired_size / min_size
        frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    elif isinstance(desired_size, tuple):
        frame = cv2.resize(frame, dsize=(desired_size[1], desired_size[0]), interpolation=cv2.INTER_CUBIC)
    return frame


def random_temporal_crop(tensor, min_size):
    while tensor.shape[0] < min_size:
        tensor = np.concatenate([tensor, tensor], 0)
    offset = np.random.randint(max(len(tensor) - min_size, 1))
    return tensor[offset:offset + min_size]


def repeat_tensor(tensor, repeat_times=None, min_size=None, axis=0, segments=None):
    if repeat_times is None:
        repeat_times = 1
        while tensor.shape[axis] <= min_size:
            if segments is not None:
                if axis == 0:
                    q_len, r_len = tensor.shape[axis], 0
                elif axis == 1:
                    q_len, r_len = 0, tensor.shape[axis]
                for (q_min, r_min, q_max, r_max) in list(segments):
                    segments.append([q_min+q_len, r_min+r_len, q_max+q_len, r_max+r_len])
            tensor = np.concatenate([tensor, tensor], axis)
            repeat_times *= 2
    else:
        tensor = np.concatenate([tensor]*repeat_times, axis)
    return tensor, repeat_times


def load_video_ffmpeg(video, start=None, end=None, fps=None, crop=None, resize=None):
    probe = ffmpeg.probe(video)
    video_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    if start is not None and end is not None:
        cap = ffmpeg.input(video, ss=start, to=end)
    else:
        cap = ffmpeg.input(video)
        
    if fps is not None:
        cap = cap.filter('fps', fps=fps)
    
    if isinstance(resize, int):
        min_size = np.min([width, height])
        ratio = resize / min_size
        height = int(np.ceil(height * ratio / 2) * 2)
        width = int(np.ceil(width * ratio / 2) * 2)
        cap = cap.filter('scale', width=width, height=height)
    elif isinstance(resize, tuple):
        height = resize[0]
        width = resize[1]
        cap = cap.filter('scale', width=resize[1], height=resize[0])
        
    if isinstance(crop, int):
        y = int(np.maximum(0, (height - crop)/2))
        x = int(np.maximum(0, (width - crop)/2))
        cap = cap.filter('crop', x=x, y=y, w=crop, h=crop)
        height = crop
        width = crop
    elif isinstance(crop, tuple):
        y = int(np.maximum(0, (height - crop[0])/2))
        x = int(np.maximum(0, (width - crop[1])/2))
        cap = cap.filter('crop', x=x, y=y, w=crop[1], h=crop[0])
        height = crop[0]
        width = crop[1]

    out, err = cap.output('pipe:', format='rawvideo', pix_fmt='rgb24', crf=0).global_args(
        '-loglevel', 'panic').run(capture_stdout=True)

    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return video


def load_video_opencv(video, all_frames=False, fps=1, crop=None, resize=None):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    if fps_v > 144 or fps_v is None:
        fps_v = 25
    frames = []
    count = 0
    while cap.isOpened():
        _ = cap.grab()
        if int(count % round(fps_v / fps)) == 0 or all_frames:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if resize is not None:
                    frame = resize_frame(frame, resize)
                frames.append(frame)
            else:
                break
        count += 1
    cap.release()
    frames = np.array(frames)
    if crop is not None:
        frames = center_crop(frames, crop)
    return frames


def load_frames_opencv(video_dir, start=0, end=None, crop=None, resize=None):
    cv2.setNumThreads(2)
    if end is None: end = len(os.listdir(video_dir))
    
    frames = []
    for frame_id in range(start, end):
        frame_file = os.path.join(video_dir, f'{frame_id:05d}.jpg')
        if os.path.exists(frame_file):
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = resize_frame(frame, resize)
            frames.append(frame)
    assert len(frames) > 0, '{} {} {}'.format(video_dir, start, end)
    frames = np.stack(frames)
    if crop is not None:
        frames = center_crop(frames, crop)
    return frames


def get_video_length(video_dir):
    return len(os.listdir(video_dir))


def load_video(video_id, video_dir=None, fps=1, start=0, end=None, window=None, repeat_times=1, resize=None, crop=None):

    if video_dir is not None:
        video_id = os.path.join(video_dir, video_id)

    video_file = glob.glob(os.path.join(video_id, 'video.*'))[0]
    if repeat_times > 1:
        video = load_video_ffmpeg(video_file, fps=fps, resize=resize, crop=crop)
        video, _ = repeat_tensor(video, repeat_times)
        video = video[start: end]
    else:
        if window is not None:
            video_len = get_video_length(video_file)
            if video_len > window:
                start = np.random.randint(max(video_len - window, 1))
                end = start + window
        video = load_video_ffmpeg(video_file, start=start, end=end, fps=fps, resize=resize, crop=crop)
        if window is not None:
            video = random_temporal_crop(video, window)
    return video


def load_frames(video_id='', video_dir=None, start=0, end=None, window=None, repeat_times=1, resize=256, crop=None):

    if video_dir is not None:
        video_id = os.path.join(video_dir, video_id)

    if repeat_times > 1:
        video = load_frames_opencv(video_id, resize=resize, crop=crop)
        video, _ = repeat_tensor(video, repeat_times)
        video = video[start: end]
    else:
        if window is not None and start == 0 and end is None:
            video_len = get_video_length(video_id)
            if video_len > window:
                start = np.random.randint(max(video_len - window, 1))
                end = start + window
        video = load_frames_opencv(video_id, start=start, end=end, resize=resize, crop=crop)
        if window is not None:
            video = random_temporal_crop(video, window)
    return video


def load_features(feature_file, video_id, start=0, end=None, repeat_times=1):
    features = feature_file[video_id][:]
    feature, _ = repeat_tensor(features, repeat_times)
    feature = feature[start: end]
    return feature


def heatmap(sim, vmin=None, vmax=None):
    ax = sns.heatmap(sim, cmap="jet", square=True, vmin=vmin, vmax=vmax, yticklabels=False, xticklabels=False)
    plt.tight_layout()

    io_buf = BytesIO.BytesIO()
    ax.figure.savefig(io_buf, format='raw', pad_inches=0)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(ax.figure.bbox.bounds[3]), int(ax.figure.bbox.bounds[2]), -1))
    io_buf.close()
    plt.clf()
    return img_arr[:, :, :3]


@torch.no_grad()
def writer_log(writer, model, meters, total_values, lr, videos, features, global_step):
    model.eval()
    for k, v in meters.items():
        writer.add_scalar('training/{}'.format(k), v.avg(total_values), global_step)
    for k, v in model.state_dict().items():
        writer.add_histogram(str(k).replace('.', '/'), v, global_step)

    writer.add_scalar('training/lr', lr, global_step)

    if hasattr(model, 'attention'):
        writer.add_histogram('att/weights', model.attention(features)[1], global_step)
    
    features = model.index_video(features)
    writer.add_histogram('features', features, global_step)
    
    if hasattr(model, 'similarity_matrix'):
        idx = np.random.randint(videos.shape[0] // 2)

        anchors, positives = torch.chunk(features, 2, dim=0)
        sim_out, sim_in, _ = model.similarity_matrix(
            anchors[idx], positives[idx], return_f2f=True, normalize=True, batched=True)

        sim_out = sim_out.cpu().numpy()
        sim_in = sim_in.cpu().numpy()

        a, p = np.unravel_index(sim_in[0, 0].argmax(), sim_in[0, 0].shape)

        writer.add_image('frames/anchor', videos[idx, a].cpu(), global_step, dataformats='HWC')
        writer.add_image('frames/positive', videos[idx + videos.shape[0] // 2, p].cpu(), global_step, dataformats='HWC')

        writer.add_image('similarity_matrices/input_matrix', heatmap(sim_in[0].mean(0)),
                         global_step, dataformats='HWC')
        writer.add_image('similarity_matrices/output_matrix', heatmap(sim_out[0, 0], 0., 1.),
                         global_step, dataformats='HWC')
    torch.cuda.empty_cache()
    model.train()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.3f'):
        self.name = name
        self.fmt = fmt
        self.values = []

    def reset(self):
        self.values = []

    def update(self, val):
        self.values.append(val)

    def avg(self, n=None):
        avg = self.values[-n:] if n is not None else self.values
        return np.mean(avg)

    def last(self):
        return self.values[-1]

    def __len__(self):
        return len(self.values)

    def __str__(self):
        fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(val=self.last(), avg=self.avg())


class AverageMeterDict(object):
    def __init__(self):
        self.meter_dict = dict()

    def reset(self):
        for k, v in self.meter_dict.items():
            v.reset()

    def add(self, name, fmt=':.3f'):
        self.meter_dict[name] = AverageMeter(name, fmt)

    def get(self, name):
        return self.meter_dict[name]

    def update(self, name, val):
        if isinstance(val, torch.Tensor):
            val = val.clone().detach().cpu().numpy()
        if name not in self.meter_dict:
            self.add(name)
        self.meter_dict[name].update(val)

    def avg(self, n=None):
        return {k: v.avg(n) for k, v in self.meter_dict.items()}

    def last(self):
        return {k: v.last() for k, v in self.meter_dict.items()}

    def items(self):
        return self.meter_dict.items()

    def to_str(self):
        return {k: str(v) for k, v in self.meter_dict.items()}

    def __len__(self):
        return min([len(v) for v in self.meter_dict.values()])
