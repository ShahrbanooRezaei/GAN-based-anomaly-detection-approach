""" 
Modified by Shahrbanoo Rezaei,  01/01/2021
Source code: "https://github.com/samet-akcay/ganomaly/tree/master/lib"

Using GAnomaly for Anomaly Detection in CAV
"""

""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='CAV_reviewer_run', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=256, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--isize1', type=int, default=4, help='input image size.')
        self.parser.add_argument('--isize2', type=int, default=245, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=1, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=32)
        self.parser.add_argument('--ndf', type=int, default=32)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='ganomaly', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of anomalies in test set.')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')
        ### SC referes to MaxAbsScaler

        ## 
        #
        #self.parser.add_argument('--VAL_dataset_root', default='data/l_f_GM_state_train_190k.csv', help='path to dataset LL') #
        self.parser.add_argument('--Train_dataset_root', default='g2_real_data_row.csv', help='path to dataset LL')
        self.parser.add_argument('--n', type=int, default=7350000, help='training size')
        #self.parser.add_argument('--nv', type=int, default=15000, help='validation size')
        self.parser.add_argument('--Test_dataset_root', default='test5_all.csv', help='path to dataset')
        self.parser.add_argument('--label_root', default='test5_anomaly_label_all.csv', help='label root')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--display_train_img', type=int, default=100, help='frequency of displaying real and fake images')
        self.parser.add_argument('--display_test_img', type=int, default=64, help='frequency of displaying real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', default=1, action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for') #220
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Adversarial loss weight')
        self.parser.add_argument('--w_con', type=float, default=50, help='Reconstruction loss weight')
        self.parser.add_argument('--w_enc', type=float, default=1, help='Encoder loss weight.')
        self.parser.add_argument('--w_kld', type=float, default=1, help='KLD loss weight.')
        self.parser.add_argument('--result_name', type=float, default=['Score_LS1', 'Score_LS2', 'Score_FS1', 'Score_FS2'], help='Result files name')
        self.parser.add_argument('--tun_name', type=str, default='test0.5', help='Tunning') 
        self.parser.add_argument('--seq_sp', type=int, default=245, help='sequence step for training')
        self.parser.add_argument('--per_weight', type=float, default=[0,0,0,0,0,1], help='score weights')
        self.parser.add_argument('--threshold', type=float, default=[0,0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], help='Thresholds')
        self.parser.add_argument('--scal', type=int, default=0, help='if test/val use train scaler (0) or not (1)')
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join('experiments',self.opt.dataset, self.opt.tun_name)
        # test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        # if not os.path.isdir(test_dir):
        #     os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
