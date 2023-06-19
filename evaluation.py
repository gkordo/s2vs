import torch
import utils
import argparse
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import EvaluationDataset
from model.similarity_network import SimilarityNetwork
from model.feature_extractor import FeatureExtractor
from datasets.generators import VideoDatasetGenerator, HDF5DatasetGenerator


@torch.no_grad()
def extract_features(feat_ext, video, batch_sz=1024, gpu_id=0, min_len=4):
    with torch.cuda.amp.autocast():
        features = [feat_ext(batch.to(gpu_id)) for batch in utils.batching(video, batch_sz)]
        features = torch.cat(features, 0)
        while features.shape[0] < min_len:
            features = torch.cat([features, features], 0)
    return features


@torch.no_grad()
def calculate_similarities_to_queries(model, queries, target, args):
    similarities = []
    batch_sz = 2048 if 'batch_sz_sim' not in args else args.batch_sz_sim
    for i, query in enumerate(queries):
        if query.device.type == 'cpu':
            query = query.to(args.gpu_id)
        sim = []
        for batch in utils.batching(target, batch_sz):
            sim.append(model.calculate_video_similarity(query, batch, apply_visil=not args.features_only))
        sim = torch.mean(torch.cat(sim, 0))
        similarities.append(sim.cpu().numpy())
    return similarities 
    
    
@torch.no_grad()
def query_vs_target(feat_extractor, sim_network, dataset, args, verbose=True):

    # Create a video generator for the queries
    if args.use_features:
        generator = HDF5DatasetGenerator(args.dataset_hdf5, dataset.get_queries())
    else:
        generator = VideoDatasetGenerator(args.dataset_path, dataset.get_queries(), args.pattern,
                                          loader=args.loader, fps=args.fps, crop=args.crop, resize=args.resize)
    loader = DataLoader(generator, num_workers=args.workers)

    # Extract features of the queries
    all_db, queries, queries_ids = set(), [], []
    if verbose: print('\n> Extract features of the query videos')
    pbar = tqdm(loader) if verbose else loader
    for (video_tensor,), (video_id,) in pbar:
        if video_id and video_tensor.shape[0]:
            if video_tensor.ndim > 3:
                features = extract_features(feat_extractor, video_tensor, args.batch_sz, args.gpus[0])
            else:
                features = video_tensor.to(args.gpus[0])
            if not args.features_only: features = sim_network.index_video(features)
            if not args.load_queries: features = features.cpu()
            all_db.add(video_id)
            queries.append(features)
            queries_ids.append(video_id)
            if verbose:
                if video_tensor.ndim > 3:
                    pbar.set_postfix(query=video_id, features=features.shape, video=video_tensor.shape)
                else:
                    pbar.set_postfix(query=video_id, features=features.shape)

    # Create a video generator for the database video
    if args.use_features:
        generator = HDF5DatasetGenerator(args.dataset_hdf5, dataset.get_database())
    else:
        generator = VideoDatasetGenerator(args.dataset_path, dataset.get_database(), args.pattern,
                                          loader=args.loader, fps=args.fps, crop=args.crop, resize=args.resize)
    loader = DataLoader(generator, num_workers=args.workers)
    
    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    if verbose: print('\n> Calculate query-target similarities')
    pbar = tqdm(loader) if verbose else loader
    for (video_tensor,), (video_id,) in pbar:
        if video_id and video_tensor.shape[0]:
            if video_tensor.ndim > 3:
                features = extract_features(feat_extractor, video_tensor, args.batch_sz, args.gpus[0])
            else:
                features = video_tensor.to(args.gpus[0])
            if not args.features_only: features = sim_network.index_video(features)
            sims = calculate_similarities_to_queries(sim_network, queries, features, args)
            all_db.add(video_id)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
            if verbose:
                if video_tensor.ndim > 3:
                    pbar.set_postfix(target=video_id, features=features.shape, video=video_tensor.shape)
                else:
                    pbar.set_postfix(target=video_id, features=features.shape)

    if args.store_similarities:
        import pickle as pk
        with open('results/{}_similarities.pk'.format(dataset.name.lower()), 'wb') as f:
            pk.dump(similarities, f)
    if verbose: print('\n> Evaluation on {}'.format(dataset.name))
    return dataset.evaluate(similarities, all_db, verbose=verbose)

    
if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='This is the code for the evaluation of the trained student on five datasets.',
        formatter_class=formatter)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "EVVE", "VCDB"],
                        help='Name of evaluation dataset.')
    parser.add_argument('--dataset_hdf5', type=str, default=None,
                        help='Path to hdf5 file containing the features of the evaluation dataset.')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to file that contains the database videos.')
    parser.add_argument('--distractors', type=utils.bool_flag, default=False,
                        help='Path to hdf5 file containing the features of the evaluation dataset.')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Pattern that the videos are stored in the video directory, eg. \"{id}/video.*\" '
                             'where the \"{id}\" is replaced with the video Id. Also, it supports '
                             'Unix style pathname pattern expansion.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a trained student network. If it is not provided, '
                             'then the pretrained weights are used with the default architecture.')
    parser.add_argument('--pretrained', type=str, default='s2vs_dns', choices=['s2vs_dns', 's2vs_vcdb'],
                        help='Pretrained network that will be used for similarity calculation.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers used for video loading.')
    parser.add_argument('--backbone', type=str, default='resnet', choices=[x.name.lower() for x in FeatureExtractor],
                        help='Backbone network used for feature extraction.')
    parser.add_argument('--dims', type=int, default=512,
                        help='Dimensionality of the input features.')
    parser.add_argument('--batch_sz', type=int, default=300,
                        help='Number of frames processed in each batch.')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='Number of feature tensors in each batch during similarity calculation.')
    parser.add_argument('--features_only', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating whether symmetric similarity matrices will be computed.')
    parser.add_argument('--fps', type=int, default=1,
                        help='Fps value for video loading.')
    parser.add_argument('--crop', type=int, default=224,
                        help='Crop value for video loading.')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize value for video loading.')
    parser.add_argument('--load_queries', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether the query features will be loaded to the GPU memory.')
    parser.add_argument('--loader', type=str, default='video', choices=['video', 'frame'],
                        help='Format of the videos stored in the dataset path. Use \'video\' is videos are '
                             'stored in format .mp4, .webm, .flv etc. Use \'frame\' if you have extracted video frames')
    parser.add_argument('--store_similarities', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating whether the output similarities will be stored.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='ID of the GPU used for the student evaluation. Comma-separated to use more than one '
                             'GPU during feature extraction.')
    args = parser.parse_args()

    dataset = EvaluationDataset[args.dataset.upper().replace('-', '_')].get_dataset(args.distractors)

    args.gpus = list(map(int, args.gpu_id.split(',')))
    args.use_features = args.dataset_hdf5 is not None

    feat_extractor = None
    if args.model_path is not None:
        print('\n> Loading network')
        d = torch.load(args.model_path, map_location='cpu')
        model_args = d['args']
        if not args.use_features:
            feat_extractor = FeatureExtractor[model_args.backbone.upper()].get_model(model_args.dims)
        sim_network = SimilarityNetwork[d['args'].similarity_network].get_model(**vars(d['args']))
        sim_network.load_state_dict(d['model'])
    else:
        if not args.use_features:
            feat_extractor = FeatureExtractor[args.backbone.upper()].get_model(args.dims)
        sim_network = SimilarityNetwork['ViSiL'].get_model(pretrained=args.pretrained)

    if not args.use_features:
        feat_extractor = nn.DataParallel(feat_extractor, device_ids=args.gpus).to(args.gpus[0]).eval()
    sim_network = sim_network.to(args.gpus[0]).eval()
    print(sim_network)

    query_vs_target(feat_extractor, sim_network, dataset, args)
