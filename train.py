"""
TRAIN AND TEST GANOMALY ON CAV DATA (data from car following model paper)
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_Train_data
from lib.model import Ganomaly
import os
from lib.evaluate_Test import performance

#
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##add path
    direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'Train')
    if not os.path.exists(direct):
            os.makedirs(direct)
    direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'Test')
    if not os.path.exists(direct):
            os.makedirs(direct)
    direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'performance')
    if not os.path.exists(direct):
            os.makedirs(direct)
    direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'performance_train')
    if not os.path.exists(direct):
            os.makedirs(direct)
    ##
    # LOAD DATA
    dataloader, scaler = load_Train_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader, scaler)
    ##
    # TRAIN MODEL
    model.train()

if __name__ == '__main__':
    train()

"""    ##########################run##########################        """

# opt = Options().parse()
# ##add path
# direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'Train')
# if not os.path.exists(direct):
#         os.makedirs(direct)
# direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'Test')
# if not os.path.exists(direct):
#         os.makedirs(direct)
# direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'performance')
# if not os.path.exists(direct):
#         os.makedirs(direct)
# direct = os.path.join('experiments',opt.dataset, opt.tun_name, 'performance_train')
# if not os.path.exists(direct):
#         os.makedirs(direct)
# ##
# # LOAD DATA
# dataloader, scaler = load_Train_data(opt)

# model = Ganomaly(opt, dataloader, scaler)

# from lib.data import load_Test_data
# dataloader_test = load_Test_data(opt, scaler)
# model.test(dataloader_test, 'Test')