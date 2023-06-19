import h5py
import torch
import utils
import argparse
import torch.nn as nn

from tqdm import tqdm
from datasets import EvaluationDataset
from torch.utils.data import DataLoader
from datasets.generators import VideoDatasetGenerator
from model.feature_extractor import FeatureExtractor
    
    
@torch.no_grad()
def extract_features(feat_extractor, dataset, args):

    videos = sorted(list(set(dataset.get_queries() + dataset.get_database())))
    if args.square:
        args.resize = (args.resize, args.resize)
    # Create a video generator for the queries
    generator = VideoDatasetGenerator(args.dataset_path, videos, args.pattern, loader=args.loader,
                                      fps=args.fps, crop=args.crop, resize=args.resize)
    loader = DataLoader(generator, num_workers=args.workers)

    # Extract features of the queries
    print('\n> Extract features of the query videos')
    pbar = tqdm(loader)
    with h5py.File(args.dataset_hdf5, "w") as hdf5_file:
        for ((video_tensor,), (video_id,)) in pbar:
            if video_id and video_tensor.shape[0] > 0:
                with torch.cuda.amp.autocast():
                    features = [feat_extractor(batch) for batch in utils.batching(video_tensor, args.batch_sz)]
                    features = torch.cat(features, 0)
                    features = features.cpu().numpy()
                hdf5_file.create_dataset(video_id, data=features, dtype="f", compression='gzip', compression_opts=9)
                pbar.set_postfix(query=video_id, features=features.shape, video=video_tensor.shape)

    
if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='This is the code for feature extraction on six datasets.', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["FIVR-200K", "FIVR-5K", "DnS-100K", "CC_WEB_VIDEO", "EVVE", "VCDB"],
                        help='Name of evaluation dataset.')
    parser.add_argument('--dataset_hdf5', type=str, required=True,
                        help='Path to hdf5 file containing the features of the evaluation dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to file that contains the database videos')
    parser.add_argument('--pattern', type=str, required=True,
                        help='Pattern that the videos are stored in the video directory, eg. \"{id}/video.*\" '
                             'where the \"{id}\" is replaced with the video Id. Also, it supports '
                             'Unix style pathname pattern expansion.')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of workers used for video loading.')
    parser.add_argument('--backbone', type=str, default='resnet', choices=[x.name.lower() for x in FeatureExtractor],
                        help='Backbone network used for feature extraction.')
    parser.add_argument('--dims', type=int, default=512,
                        help='Dimensionality of the input features.')
    parser.add_argument('--batch_sz', type=int, default=300,
                        help='Number of frames processed in each batch.')
    parser.add_argument('--fps', type=int, default=1,
                        help='Fps value for video loading.')
    parser.add_argument('--crop', type=int, default=224,
                        help='Crop value for video loading.')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize value for video loading.')
    parser.add_argument('--square', type=utils.bool_flag, default=False,
                        help='Resize value for video loading.')
    parser.add_argument('--loader', type=str, default='video', choices=['video', 'frame'],
                        help='Format of the videos stored in the dataset path. Use \'video\' is videos are '
                             'stored in format .mp4, .webm, .flv etc. Use \'frame\' if you have extracted video frames')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='ID of the GPU used for the student evaluation. Comma-separated to use more than one '
                             'GPU during feature extraction.')
    args = parser.parse_args()

    dataset = EvaluationDataset[args.dataset.upper().replace('-', '_')].get_dataset()

    args.gpus = list(map(int, args.gpu_id.split(',')))

    feat_extractor = FeatureExtractor[args.backbone.upper()].get_model(args.dims)
    feat_extractor = nn.DataParallel(feat_extractor, device_ids=args.gpus).to(args.gpus[0]).eval()

    extract_features(feat_extractor, dataset, args)
