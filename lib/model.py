""" 
Modified by Shahrbanoo Rezaei,  01/01/2021
Source code: "https://github.com/samet-akcay/ganomaly/tree/master/lib"

Using GAnomaly for Anomaly Detection in CAV
"""

"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import pandas as pd

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.data import load_Test_data
from lib.data import load_Train_data_2
from lib.data import load_val_data
from lib.evaluate_Train import EV_T
from lib.evaluate_Test import performance
from lib.evaluate_Test import performance_rev


class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader, scaler):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.scaler = scaler
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.dataloader_test = load_Test_data(self.opt, self.scaler)
        self.dataloader_TrTe = load_Train_data_2(self.opt, self.scaler)
        self.dataloader_val = load_val_data(self.opt, self.scaler)

    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())
            
            
            

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """
        
        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader, leave=False, total=len(self.dataloader)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.display_train_img == 0:
                reals, fakes, fixed = self.get_current_images()
                #self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                #if self.opt.display:
                mode = 'Train'
                # self.visualizer.display_current_images_dif(reals, fakes, fixed, mode, self.epoch, self.epoch)
                #self.visualizer.display_current_images(reals, fakes, fixed, mode, self.epoch, self.epoch)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        
        
        self.loss_rev = []
            
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.loss_d_real_rev = []
            self.loss_d_fake_rev = []
            self.loss_g_rev = []
            self.acc_real_rev = []
            self.acc_fake_rev = []
            
            
            # print(self.err_g)
            self.train_one_epoch()
            
            l = len(self.loss_d_real_rev)
            self.loss_rev.append([sum(self.loss_d_real_rev)/l,sum(self.loss_d_fake_rev)/l,sum(self.loss_g_rev)/l,sum(self.acc_real_rev)/l, sum(self.acc_fake_rev)/l])

            ############### Reviewer
            self.test(self.dataloader_test,'reviewer')
            
            
            ###############
            
            if self.epoch % 5 ==0 :
                # self.test(self.dataloader_VAL,'VAL')
                # self.test(self.dataloader_test,'Test')
                # self.test(self.dataloader_val,'val')
                # self.test(self.dataloader_test_MW,'Test_MW')
                self.save_weights(self.epoch)
        
        df3=pd.DataFrame(self.loss_rev)
        df3.columns=['loss_d_real_rev','loss_d_fake_rev', 'loss_g_rev', 'acc_real_rev', 'acc_fake_rev']
        df3.to_csv('experiments\{:s}\{:s}\performance\{:d}_loss.csv'.format(self.opt.dataset,self.opt.tun_name,self.epoch), index=False)
    
        # self.test(self.dataloader_test,'Test')
        # self.test(self.dataloader_test_MW,'Test_MW')
        # self.test(self.dataloader_TrTe,'Train_Test')
        self.save_weights(self.epoch)
        
        print(">> Training model %s.[Done]" % self.name)
        
    ##
    def test(self, dataloader_t, phase):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')
                
            #n_B = len(dataloader_t) * self.opt.batchsize  
            n_B = len(dataloader_t.dataset) ##if last drop is False
            scores = torch.zeros(size=(self.opt.isize1, n_B*self.opt.isize2,6), dtype=torch.float32, device=self.device)
            scor = torch.zeros(size=( n_B*self.opt.isize2,self.opt.isize1*2), dtype=torch.float32, device=self.device)
            
            print(scores.size())
            
            print(" %s model %s." % (phase, self.name))
           
            self.times = []
            # self.total_steps = 0
            # epoch_iter = 0
            bb=0
            for i, data in enumerate(dataloader_t, 0):
                # self.total_steps += self.opt.batchsize
                # epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                
                self.fake, latent_i, latent_o = self.netg(self.input)
                # print(self.fake.shape)
                pred_test, feat_test = self.netd(self.input)
                pred_fake, feat_fake = self.netd(self.fake)
                
                err_g_adv = torch.mean(torch.abs((self.netd(self.input)[1]-self.netd(self.fake)[1])), dim=(1,2,3)) #!!!!!
                
                err_g_enc = torch.mean(torch.pow((latent_i-latent_o), 2), dim=(1,2,3))
                
                # print(self.input)
                time_o = time.time()
                
                for k in range(self.opt.isize1):
                    b=-1
                    for jj in range(i*self.opt.batchsize,i*self.opt.batchsize+err_g_adv.size(0)):
                        b=b+1
                        scores[k,jj*self.opt.isize2 : (jj+1)*self.opt.isize2,0]=err_g_adv[b]
                        scores[k,jj*self.opt.isize2 : (jj+1)*self.opt.isize2,1]=err_g_enc[b]
                        scores[k,jj*self.opt.isize2 : (jj+1)*self.opt.isize2,2]=pred_test[b]
                        scores[k,jj*self.opt.isize2 : (jj+1)*self.opt.isize2,3]=pred_fake[b]
                        scores[k,jj*self.opt.isize2 : (jj+1)*self.opt.isize2,4]=(self.input[b,0,k,:]-self.fake[b,0,k,:])**2
                        scores[k,jj*self.opt.isize2 : (jj+1)*self.opt.isize2,5]=torch.abs(self.input[b,0,k,:]-self.fake[b,0,k,:])
                        scor[jj*self.opt.isize2 : (jj+1)*self.opt.isize2,k] = self.input[b,0,k,:]
                        scor[jj*self.opt.isize2 : (jj+1)*self.opt.isize2,k+4] = self.fake[b,0,k,:]
                self.times.append(time_o - time_i)
                
                
                reals, fakes, fixed = self.get_current_images()
                if phase=='Test':
                    mode = 'Test'
                    self.visualizer.display_current_images_dif(reals, fakes, fixed, mode, self.epoch,i)
                if bb == 0:
                    if phase=='val':
                        mode = 'val'
                        self.visualizer.display_current_images_dif3(reals, fakes, fixed, mode, self.epoch,i)
                bb=1

                

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)
            
            
            if phase =='Test':
                for jj in range(self.opt.isize1):
                    n= self.opt.result_name
                    np.savetxt('experiments/%s/%s/performance/fe_%d_%s.csv' % (self.opt.dataset,self.opt.tun_name,self.epoch, n[jj]), scores.squeeze().cpu().numpy()[jj,:,:], delimiter=',')
                np.savetxt('experiments/%s/%s/performance/fe_%d_NM.csv' % (self.opt.dataset,self.opt.tun_name,self.epoch), scor.squeeze().cpu().numpy(), delimiter=',')
                performance(self.opt,self.epoch)
            
            if phase =='reviewer':
                for jj in range(self.opt.isize1):
                    n= self.opt.result_name
                    np.savetxt('experiments/%s/%s/performance/fe_%d_%s.csv' % (self.opt.dataset,self.opt.tun_name,0, n[jj]), scores.squeeze().cpu().numpy()[jj,:,:], delimiter=',')
                np.savetxt('experiments/%s/%s/performance/fe_%d_NM.csv' % (self.opt.dataset,self.opt.tun_name,0), scor.squeeze().cpu().numpy(), delimiter=',')
                performance_rev(self.opt,0,self.epoch)
            
            if phase =='val':
                for jj in range(self.opt.isize1):
                    n= self.opt.result_name
                    np.savetxt('experiments/%s/%s/performance/fe_%d_%s.csv' % (self.opt.dataset,self.opt.tun_name,self.epoch, n[jj]), scores.squeeze().cpu().numpy()[jj,:,:], delimiter=',')
                np.savetxt('experiments/%s/%s/performance/fe_%d_NM.csv' % (self.opt.dataset,self.opt.tun_name,self.epoch), scor.squeeze().cpu().numpy(), delimiter=',')
                   
            
            
            if phase=='Train_Test':
                EV_T(self.epoch,scores.squeeze().cpu().numpy(),self.opt,phase).performance()
                np.savetxt('experiments/%s/%s/performance_train/fe_TrTe_%d_NM.csv' % (self.opt.dataset,self.opt.tun_name,self.epoch), scor.squeeze().cpu().numpy(), delimiter=',')
                 
##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader, scaler):
        super(Ganomaly, self).__init__(opt, dataloader, scaler)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        
        
        

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize1, self.opt.isize2), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize1, self.opt.isize2), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())
        
        

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)
        
        
        self.loss_g_rev.append(self.err_g.item())
        

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()
        
        self.loss_d_real_rev.append(self.err_d_real.item())
        self.loss_d_fake_rev.append(self.err_d_fake.item())
        
        self.acc_real_rev.append(torch.mean(self.pred_real).item())
        self.acc_fake_rev.append(torch.mean(self.pred_fake).item())
        

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()

