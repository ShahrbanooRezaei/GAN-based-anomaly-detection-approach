""" 
Modified by Shahrbanoo Rezaei,  01/01/2021
Source code: "https://github.com/samet-akcay/ganomaly/tree/master/lib"

Using GAnomaly for Anomaly Detection in CAV
"""

""" Network architectures 4, 120.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize1, isize2, nz, nc, ndf, ngpu, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        

        main = nn.Sequential()
        # input is nc x isize1 x isize2 (1 x 4 x 245)
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, (1,82), (1,5), (1,1), bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        
        # state size. (ndf) x 6 x 34
        
        main.add_module('pyramid-{0}-{1}-conv'.format(ndf, ndf*2),
                        nn.Conv2d(ndf, ndf*2, (3,18), (1,6), (0,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format( ndf*2),
                        nn.BatchNorm2d( ndf*2))
        main.add_module('pyramid-{0}-relu'.format( ndf*2),
                        nn.LeakyReLU(0.2, inplace=True))
        
        
        # state size. (ndf*8) x 4 x 4
        
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(ndf*2, 1),
                            nn.Conv2d(ndf*2, nz, 4, 1, 0, bias=False))
            
            #main.sigmoid()

        self.main = main
        
    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

from torchsummary import summary
#print model summary >Check Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
model = Encoder(4, 245, 100, 1, 32, 1).to(device)
summary(model, (1, 4, 245))
#print(model)

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize1, isize2, nz, nc, ngf, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        
        
        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, ngf*2),
                        nn.ConvTranspose2d(nz, ngf*2, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(ngf*2),
                        nn.BatchNorm2d(ngf*2))
        main.add_module('initial-{0}-relu'.format(ngf*2),
                        nn.ReLU(True))

        
        main.add_module('pyramid-{0}-{1}-convt'.format(ngf*2, ngf),
                        nn.ConvTranspose2d(ngf*2, ngf ,(3,18), (1,6), (0,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(ngf),
                        nn.BatchNorm2d(ngf))
        main.add_module('pyramid-{0}-relu'.format(ngf),
                            nn.ReLU(True))
        

        main.add_module('final-{0}-{1}-convt'.format(ngf, nc),
                        nn.ConvTranspose2d(ngf, nc, (1,82), (1,5), (1,1), bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

#print model summary >Check Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
model = Decoder(4, 245, 1, 1, 32, 1).to(device)
summary(model, (1, 1, 1))
#print(model)



##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.isize1, opt.isize2, 1, opt.nc, opt.ngf, opt.ngpu)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.isize1, opt.isize2, opt.nz, opt.nc, opt.ngf, opt.ngpu)
        self.decoder = Decoder(opt.isize1, opt.isize2, opt.nz, opt.nc, opt.ngf, opt.ngpu)
        self.encoder2 = Encoder(opt.isize1, opt.isize2, opt.nz, opt.nc, opt.ngf, opt.ngpu)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o


