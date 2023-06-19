import utils
import einops
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import augmentations
from datasets.generators import SSLGenerator
from model.losses import InfoNCELoss, SSHNLoss
from model.similarity_network import SimilarityNetwork
from model.feature_extractor import FeatureExtractor


@torch.no_grad()
def extract_features(feat_ext, videos, batch_sz=512):
    # Feature extraction process
    b, f = videos.shape[:2]
    videos = einops.rearrange(videos, 'b f c h w -> (b f) c h w')
    features = [feat_ext(batch.cuda()) for batch in utils.batching(videos, batch_sz)]
    features = torch.cat(features, 0)
    features = einops.rearrange(features, '(b f) r d -> b f r d', b=b)
    return features


def main(args):
    # Initialization of  distributed processing
    utils.init_distributed_mode(args)
    utils.pprint_args(args)

    print('\n> Create augmentations: {}'.format(args.augmentations))
    # Instantiation of the objects for weak and strong augmentations
    weak_aug = augmentations.WeakAugmentations(**vars(args))
    strong_aug = augmentations.StrongAugmentations(**vars(args))
    print(*[weak_aug, strong_aug], sep='\n')

    # Initialization of data generator and data loader
    print('\n> Create generator')
    dataset = SSLGenerator(weak_aug=weak_aug, strong_aug=strong_aug, **vars(args))
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, num_workers=args.workers, batch_size=args.batch_sz,
                        collate_fn=dataset.collate_fn, sampler=sampler)
    args.epochs = int(args.iter_epochs // len(loader) + 1)

    # Initialization of the feature extraction network and similarity network
    print('\n> Building network')
    feat_extractor = FeatureExtractor[args.backbone.upper()].get_model(args.dims).cuda().eval()
    model = SimilarityNetwork[args.similarity_network].get_model(**vars(args))
    model = nn.parallel.DistributedDataParallel(model.cuda()) # only similarity network is trainable
    print(model)

    # Instantiation of our losses
    nce_criterion = InfoNCELoss(args.temperature)
    sshn_criterion = SSHNLoss()

    # Initialization of the optimizer and lr scheduler
    params = [v for v in filter(lambda p: p.requires_grad, model.parameters())]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.iter_epochs,
        lr_min=args.final_lr*args.learning_rate,
        warmup_t=args.warmup_iters,
        warmup_lr_init=args.warmup_lr_init*args.learning_rate,
        t_in_epochs=False,
    )
    global_step = 0

    # Initialization of the FP16 Scaler if used
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Load a saved model
    if args.load_model:
        global_step = utils.load_model(args, model, optimizer)

    # Initialization of the reporting tools
    meters = utils.AverageMeterDict()
    writer = SummaryWriter(args.experiment_path) if args.gpu == 0 else None

    print('\n> Start training for {} epochs'.format(args.epochs))
    # Training loop
    for epoch in range(global_step // len(loader), args.epochs):
        sampler.set_epoch(epoch)
        dataset.next_epoch()
        meters.reset()
        model.train()
        global_step = train_one_epoch(
            epoch, global_step, feat_extractor, model, loader, optimizer, lr_scheduler, fp16_scaler,
            nce_criterion, sshn_criterion, writer, meters, args)
        model.eval()

        # save model at the end of each epoch
        if args.gpu == 0:
            utils.save_model(args, model, optimizer, global_step, 'model.pth')


def train_one_epoch(epoch, global_step, extractor, model, loader, optimizer, lr_scheduler, fp16_scaler,
                    nce_criterion, sshn_criterion, writer, meters, args):

    pbar = tqdm(loader, desc='epoch {}'.format(epoch), unit='iter') if args.gpu == 0 else loader
    # Loop for the epoch
    for idx, (videos, labels) in enumerate(pbar):
        optimizer.zero_grad()
        # videos = videos.cuda()
        labels = labels.cuda()

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Extract features for the video frames
            features = extract_features(extractor, videos, args.batch_sz_fe)
            # Calculate similarities for each video pair in the batch
            similarities, regularization_loss = model(features)

            # Calculate losses
            infonce_loss = nce_criterion(similarities, labels)
            hardneg_loss, self_loss = sshn_criterion(similarities, labels)

            # Final loss
            loss = infonce_loss + args.lambda_parameter * (hardneg_loss + self_loss) + args.r_parameter * regularization_loss

        # Update model weights
        lr_scheduler.step_update(global_step)
        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        global_step += 1

        meters.update('total_loss', loss)
        meters.update('infonce_loss', infonce_loss)
        meters.update('sshn_loss', (hardneg_loss + self_loss))
        meters.update('reg_loss', regularization_loss)

        # Logging
        if args.gpu == 0:
            if global_step % 5 == 0:
                pbar.set_postfix(**meters.to_str())

            if global_step % args.log_step == 0 and len(meters) >= 10:
                utils.writer_log(writer, model.module, meters, args.log_step, optimizer.param_groups[0]['lr'],
                                 videos, features, global_step)
    return global_step


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='This is the training code of a video similarity network based on self-supervision',
        formatter_class=formatter)
    # Experiment arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to frame files of the trainset.')
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path of the experiment where the weights of the trained network and all logs will be stored.')
    parser.add_argument('--workers', default=12, type=int,
                        help='Number of workers used for the training.')
    parser.add_argument('--load_model', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating that the weights from an existing model will be loaded.')
    parser.add_argument('--log_step', type=int, default=100,
                        help='Number of steps to save logs.')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating that fp16 scaling will be used.')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html.')

    # Similarity network options
    parser.add_argument('--backbone', type=str, default='resnet', choices=[x.name.lower() for x in FeatureExtractor],
                        help='Backbone network used for feature extraction.')
    parser.add_argument('--similarity_network', type=str, default='ViSiL', choices=[x.name for x in SimilarityNetwork],
                        help='Similarity network used for similarity calculation.')
    parser.add_argument('--dims', type=int, default=512,
                        help='Dimensionality of the input features.')
    parser.add_argument('--attention', type=utils.bool_flag, default=True,
                        help='Boolean flag indicating whether an Attention layer will be used.')
    parser.add_argument('--binarization', type=utils.bool_flag, default=False,
                        help='Boolean flag indicating whether a Binarization layer will be used.')
    parser.add_argument('--binary_bits', type=int, default=512,
                        help='Number of bits used in the Binarization layer. Applicable only when --binarization flag is true.')

    # Training process arguments
    parser.add_argument('--augmentations', type=str, default='GT,FT,TT,ViV',
                        help='Transformations used for the strong augmentations. GT: Global Transformations '
                             'FT: Frame Transformations TT: Temporal Transformations ViV: Video-in-Video')
    parser.add_argument('--batch_sz', type=int, default=64,
                        help='Number of video pairs in each training batch.')
    parser.add_argument('--batch_sz_fe', type=int, default=512,
                        help='Number of frames in each batch for feature extraction.')
    parser.add_argument('--iter_epochs', type=int, default=30000,
                        help='Number of iterations to train the network.')
    parser.add_argument('--percentage', type=float, default=1.,
                        help='Dataset percentage used for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate used during training.')
    parser.add_argument('--final_lr', type=float, default=1e-1,
                        help='Factor based on the the base lr used for the final learning rate for the lr scheduler.')
    parser.add_argument('--warmup_iters', type=int, default=1000,
                        help='Number of warmup iterations for the lr scheduler.')
    parser.add_argument('--warmup_lr_init', type=float, default=1e-2,
                        help='Factor based on the base lr used for the initial learning rate of warmup for the lr scheduler.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay used during training.')
    parser.add_argument('--window_sz', type=int, default=32,
                        help='Number of frames of the loaded videos during training.')
    parser.add_argument('--temperature', type=float, default=0.03,
                        help='Temperature parameter for the infoNCE loss.')
    parser.add_argument('--lambda_parameter', type=float, default=3.,
                        help='Parameter that determines the impact of SSHN loss.')
    parser.add_argument('--r_parameter', type=float, default=1.,
                        help='Parameter that determines the impact of similarity regularization loss.')
    parser.add_argument('--n_raug', type=int, default=2,
                        help='Number of augmentation transformations in RandAugment. Applicable when \'GT\' is in argument \'--augmentations\'')
    parser.add_argument('--m_raug', type=int, default=9,
                        help='Magnitude for all the transformations in RandAugment. Applicable when \'GT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_overlay', type=float, default=.3,
                        help='Overlay probability in frame transformations. Applicable when \'FT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_blur', type=float, default=.5,
                        help='Blur probability in frame transformations. Applicable when \'FT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_tsd', type=float, default=.5,
                        help='Temporal Shuffle-Dropout probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_ff', type=float, default=.1,
                        help='Fast Forward probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_sm', type=float, default=.1,
                        help='Slow Motion probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_rev', type=float, default=.1,
                        help='Revision probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_pau', type=float, default=.1,
                        help='Pause probability in temporal transformations. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_shuffle', type=float, default=.5,
                        help='Shuffle probability in TSD. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_dropout', type=float, default=.3,
                        help='Dropout probability in TSD. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_content', type=float, default=.5,
                        help='Content probability in TSD. Applicable when \'TT\' is in argument \'--augmentations\'')
    parser.add_argument('--p_viv', type=float, default=.3,
                        help='Probability of applying video-in-video transformation. Applicable when \'ViV\' is in argument \'--augmentations\'')
    parser.add_argument('--lambda_viv', type=lambda x: tuple(map(float, x.split(','))), default=(.3, .7),
                        help='Resize factor range in video-in-video transformation. Applicable when \'ViV\' is in argument \'--augmentations\'')
    args = parser.parse_args()

    network_details = '{}_{}_D{}'.format(args.similarity_network.lower(), args.backbone, args.dims)
    network_details += '_att' if args.attention else ''
    network_details += '_bin_{}'.format(args.binary_bits) if args.binarization else ''

    training_details = '/ssl_{}_p{}_it{}K_W{}_t{}_lr{}_wd{}_l{}_r{}_bs{}'.format(
        args.augmentations, args.percentage, args.iter_epochs // 1000,
        args.window_sz, args.temperature, args.learning_rate, args.weight_decay,
        args.lambda_parameter, args.r_parameter, args.batch_sz)

    args.experiment_path += network_details + training_details
    main(args)
