import argparse
import torch
import os
import torch.backends.cudnn as cudnn
from datetime import datetime
from util.config import config as cfg


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str


class BaseOptions(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('--exp_name', default=cfg.get('exp_name', "meter_data"), type=str,help='Experiment name')
        self.parser.add_argument("--gpu", default=cfg.get('gpu', "0"), help="set gpu id", type=str)
        self.parser.add_argument('--resume', default=cfg.get('resume', ''), type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--num_workers', default=cfg.get('num_workers', 4), type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=cfg.get('cuda', True), type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--mgpu', action='store_true', help='Use multi-gpu to train model',default=cfg.get('mgpu', True))
        self.parser.add_argument('--save_dir', default=cfg.get('save_dir', './model/'), help='Path to save checkpoint models')
        self.parser.add_argument('--vis_dir', default=cfg.get('vis_dir', './vis/'), help='Path to save visualization images')
      
        self.parser.add_argument('--pretrain', default=cfg.get('pretrain', True), type=str2bool, help='Pretrained AutoEncoder model')
        self.parser.add_argument('--viz', action='store_true', default=cfg.get('viz', False), help='Whether to output debug info')

        # train opts
        self.parser.add_argument('--max_epoch', default=cfg.get('max_epoch', 100), type=int, help='Max epochs')
        self.parser.add_argument('--lr', '--learning-rate', default=cfg.get('lr', 1e-4), type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default=cfg.get('lr_adjust', 'fix'),
                                 choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=cfg.get('stepvalues', []), nargs='+', type=int, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', '--wd', default=cfg.get('weight_decay', 0.), type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=cfg.get('gamma', 0.1), type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=cfg.get('momentum', 0.9), type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=cfg.get('batch_size', 2), type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default=cfg.get('optim', 'Adam'), type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--save_freq', default=cfg.get('save_freq', 5), type=int, help='save weights every # epoch')
        self.parser.add_argument('--display_freq', default=cfg.get('display_freq', 10), type=int, help='display training metrics every # iter')
        self.parser.add_argument('--viz_freq', default=cfg.get('viz_freq', 50), type=int, help='visualize training process every # iter')
        self.parser.add_argument('--log_freq', default=cfg.get('log_freq', 100), type=int, help='log to tensorboard every # iterations')
        self.parser.add_argument('--val_freq', default=cfg.get('val_freq', 100), type=int, help='do validation every # iterations')

        # data set
        self.parser.add_argument('--net', default=cfg.get('net', 'vgg'), type=str,
                                 choices=['vgg', 'vgg_bn', 'resnet50', 'resnet101'],
                                 help='Network architecture')

        # data args
        self.parser.add_argument('--rescale', type=float, default=cfg.get('rescale', 255.0), help='rescale factor')
        self.parser.add_argument('--means', type=float, default=cfg.get('means', (0.485, 0.456, 0.406)), nargs='+', help='mean')
        self.parser.add_argument('--stds', type=float, default=cfg.get('stds', (0.229, 0.224, 0.225)), nargs='+', help='std')
        self.parser.add_argument('--input_size', default=cfg.get('input_size', 640), type=int, help='model input size')
        self.parser.add_argument('--test_size', default=tuple(cfg.get('test_size', (512, 1024 ))), type=tuple, help='model input size')

        # eval args
        self.parser.add_argument('--checkepoch', default=cfg.get('checkepoch', 100), type=int, help='Load checkpoint number')  #110, 220
        self.parser.add_argument('--start_epoch', default=cfg.get('start_epoch', 0), type=int, help='start epoch number')
        self.parser.add_argument('--pointer', default=cfg.get('pointer', 0.6), type=float, help='tr')
        self.parser.add_argument('--dail', default=cfg.get('dail', 0.5), type=float, help='tcl')
        self.parser.add_argument('--text', default=cfg.get('text', 0.6), type=float, help='kernel')


    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
        else:
            # torch.set_default_tensor_type('torch.FloatTensor')
            pass

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)
