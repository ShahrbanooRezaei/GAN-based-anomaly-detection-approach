""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=4
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=5
        )

    ##
    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            batch_i (int): Current batch
            batch_n (int): Total Number of batches.
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def display_current_images(self, reals, fakes, fixed, mode, epoch, bs):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        # reals = self.normalize(reals.cpu().numpy())
        # fakes = self.normalize(fakes.cpu().numpy())
        # fixed = self.normalize(fixed.cpu().numpy())

        # self.vis.images(reals, win=1, opts={'title': 'Reals'})
        # self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        # self.vis.images(fixed, win=3, opts={'title': 'Fixed'})
        
        reals = (reals.cpu().numpy())
        fakes = (fakes.cpu().numpy())
        fixed = (fixed.cpu().numpy())
        
        rand = np.random.randint(1, reals.shape[0], size=3)
        fig = plt.figure(figsize=(11,11))
        
        for j in range(3):
            ax = fig.add_subplot(3, 2, 1+2*j)
            i=rand[j]
            ax.plot((reals[i,0,0,:]),label='Leading state 1')
            ax.plot((reals[i,0,1,:]),label='Leading state 2')
            ax.plot((reals[i,0,2,:]),label='Following state 1')
            ax.plot((reals[i,0,3,:]),label='Following state 2')
            ax.legend() 
            ax.title.set_text('Real for %s th sample in batch for %s ' %(i,mode))
            
            ax = fig.add_subplot(3, 2, 2+2*j)
            i=rand[j]
            ax.plot((fakes[i,0,0,:]),label='Leading state 1')
            ax.plot((fakes[i,0,1,:]),label='Leading state 2')
            ax.plot((fakes[i,0,2,:]),label='Following state 1')
            ax.plot((fakes[i,0,3,:]),label='Following state 2')
            ax.legend() 
            ax.title.set_text('Fakes for %s th sample in batch for %s ' %(i,mode)) 
        # if mode == 'Train':
        #     plt.savefig('experiments\{:s}\{:s}\Train\{:s}_image_at_epoch_{:04d}_1.png'.format(self.opt.dataset,self.opt.tun_name,self.opt.tun_name,epoch))
        # if mode == 'Test':
        #     plt.savefig('experiments\{:s}\{:s}\Test\{:d}_image_at_batch_{:04d}_1.png'.format(self.opt.dataset,self.opt.tun_name,epoch,bs))
        plt.show()
        
    def display_current_images_dif(self, reals, fakes, fixed, mode, epoch, bs):
    
        """ Display current images.
        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        # reals = self.normalize(reals.cpu().numpy())
        # fakes = self.normalize(fakes.cpu().numpy())
        # fixed = self.normalize(fixed.cpu().numpy())

        # self.vis.images(reals, win=1, opts={'title': 'Reals'})
        # self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        # self.vis.images(fixed, win=3, opts={'title': 'Fixed'})
        
        reals = (reals.cpu().numpy())
        fakes = (fakes.cpu().numpy())
        fixed = (fixed.cpu().numpy())
        
        rand = np.random.randint(1, reals.shape[0], size=3)
        fig = plt.figure(figsize=(11,11))
        
        for j in range(3):
            ax = fig.add_subplot(3, 2, 1+2*j)
            i=rand[j]
            ax.plot((fakes[i,0,1,:]),label='Fake AV Velocity in Y')
            ax.plot((fakes[i,0,0,:]),label='Fake AV Velocity in X')
            
            
            ax.plot((reals[i,0,1,:]),label='Real AV Velocity in Y')
            ax.plot((reals[i,0,0,:]),label='Real AV Velocity in X')
            
            #ax.legend() 
            ax.title.set_text('%s th sample in %s prosess in epoch %s ' %(j+1,mode,epoch+1))
            
            ax = fig.add_subplot(3, 2, 2+2*j)
            i=rand[j]
            
            
            ax.plot((fakes[i,0,3,:]),label='Fake AG Velocity in Y')
            ax.plot((fakes[i,0,2,:]),label='Fake AG Velocity in X')
            
            
            
            ax.plot((reals[i,0,3,:]),label='Real AG Velocity in Y')
            ax.plot((reals[i,0,2,:]),label='Real AG Velocity in X')
            
            #ax.legend() 
            ax.title.set_text('%s th sample in %s prosess in epoch %s ' %(j+1,mode,epoch+1))
        if mode == 'Train':
            plt.savefig('experiments\{:s}\{:s}\Train\{:s}_image_at_epoch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,self.opt.tun_name,epoch))
        if mode == 'Test':
            plt.savefig('experiments\{:s}\{:s}\Test\{:d}_image_at_batch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,epoch,bs))
        plt.show()
        
        
    def display_current_images_dif2(self, reals, fakes, fixed, mode, epoch, bs):
    
        """ Display current images.
        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        # reals = self.normalize(reals.cpu().numpy())
        # fakes = self.normalize(fakes.cpu().numpy())
        # fixed = self.normalize(fixed.cpu().numpy())

        # self.vis.images(reals, win=1, opts={'title': 'Reals'})
        # self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        # self.vis.images(fixed, win=3, opts={'title': 'Fixed'})
        
        reals = (reals.cpu().numpy())
        fakes = (fakes.cpu().numpy())
        fixed = (fixed.cpu().numpy())
        
        # rand = np.random.randint(1, reals.shape[0], size=3)
        fig = plt.figure(figsize=(11,11))
        v = [24,0,112]
        for j in range(3):
            i = v[j]
            ax = fig.add_subplot(3,2, 1+2*j)
            
            ax.plot((fakes[i,0,1,:]), color = 'blue', label='Fake AV Velocity in Y')
            ax.plot((fakes[i,0,0,:]), color = 'blue', label='Fake AV Velocity in X')
            
            
            ax.plot((reals[i,0,1,:]), color = 'green', label='Real AV Velocity in Y')
            ax.plot((reals[i,0,0,:]), color = 'green', label='Real AV Velocity in X')
            
            
            
            ax = fig.add_subplot(3,2, 2+2*j)
            
            ax.plot((fakes[i,0,3,:]), color = 'red', label='Fake AG Velocity in Y')
            ax.plot((fakes[i,0,2,:]), color = 'red', label='Fake AG Velocity in X')
            
            ax.plot((reals[i,0,3,:]), color = 'yellow', label='Real AG Velocity in Y')
            ax.plot((reals[i,0,2,:]), color = 'yellow', label='Real AG Velocity in X')
            
            #ax.legend() 
            # ax.title.set_text('%s th sample in %s prosess in epoch %s ' %(j+1,mode,epoch+1))
            
        
        if mode == 'Train':
            plt.savefig('experiments\{:s}\{:s}\Train\{:s}_image_at_epoch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,self.opt.tun_name,epoch))
        if mode == 'Test':
            plt.savefig('experiments\{:s}\{:s}\Test\{:d}_image_at_batch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,epoch,bs))
        if mode == 'val':
            plt.savefig('experiments\{:s}\{:s}\Test\{:d}_image_at_batch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,epoch,bs))
        
        plt.show()
        
        
    def display_current_images_dif3(self, reals, fakes, fixed, mode, epoch, bs):
    
        """ Display current images.
        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        # reals = self.normalize(reals.cpu().numpy())
        # fakes = self.normalize(fakes.cpu().numpy())
        # fixed = self.normalize(fixed.cpu().numpy())

        # self.vis.images(reals, win=1, opts={'title': 'Reals'})
        # self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        # self.vis.images(fixed, win=3, opts={'title': 'Fixed'})
        
        reals = (reals.cpu().numpy())
        fakes = (fakes.cpu().numpy())
        fixed = (fixed.cpu().numpy())
        
        # rand = np.random.randint(1, reals.shape[0], size=3)
        fig = plt.figure(figsize=(11,11))
        v = [24,0,112]
        for j in range(3):
            i = v[j]
            ax = fig.add_subplot(3,2, 1+2*j)
            
            
            ax.plot((reals[i,0,1,:]), color = 'darkturquoise',linewidth=2, ls='-.', label='Real AV Velocity in Y')
            ax.plot((reals[i,0,0,:]), color = 'darkturquoise',linewidth=2,ls=':', label='Real AV Velocity in X')
            
            ax.plot((reals[i,0,3,:]), color = 'darkslategray', linewidth=2, ls='-', label='Real AG Velocity in Y')
            ax.plot((reals[i,0,2,:]), color = 'darkslategray', linewidth=2, ls='--', label='Real AG Velocity in X')
            # ax.set_xlabel('Time')
            # ax.set_ylabel('Velocity')
            
            ax = fig.add_subplot(3,2, 2+2*j)
            
            ax.plot((fakes[i,0,1,:]), color = 'darkturquoise',linewidth=2, ls='-.', label='Fake AV Velocity in Y')
            ax.plot((fakes[i,0,0,:]), color = 'darkturquoise',linewidth=2,ls=':', label='Fake AV Velocity in X')
            
            ax.plot((fakes[i,0,3,:]), color = 'darkslategray', linewidth=2, ls='-', label='Fake AG Velocity in Y')
            ax.plot((fakes[i,0,2,:]), color = 'darkslategray', linewidth=2, ls='--', label='Fake AG Velocity in X')
            # ax.set_xlabel('Time')
            # ax.set_ylabel('Velocity')
            
            
            #ax.legend() 
            # ax.title.set_text('%s th sample in %s prosess in epoch %s ' %(j+1,mode,epoch+1))
            
        
        if mode == 'Train':
            plt.savefig('experiments\{:s}\{:s}\Train\{:s}_image_at_epoch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,self.opt.tun_name,epoch))
        if mode == 'Test':
            plt.savefig('experiments\{:s}\{:s}\Test\{:d}_image_at_batch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,epoch,bs))
        if mode == 'val':
            plt.savefig('experiments\{:s}\{:s}\Test\{:d}_image_at_batch_{:04d}.png'.format(self.opt.dataset,self.opt.tun_name,epoch,bs))
        
        plt.show()
        
            
         
            
        

