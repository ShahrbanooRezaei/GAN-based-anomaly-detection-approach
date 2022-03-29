""" 
Developed by Shahrbanoo Rezaei 1/1/2021

Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
    Sensitivity, Specificity, Accuracy, F1, PPV
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

    
    
class EV_T():
    
    def __init__(self, epoch,Tscores, opt, phase):
        
        self.epoch=epoch
        self.opt = opt
        self.Tscores=Tscores
        self.phase =phase
        
    def per(self, thres,score, label,exp):
        """ Compute Accuracy, sensitivity, specificity, F1 score and PPV for different thresholds """
        CNN = [] 
        pred = []
        
        for tao in thres:
            pred = []
            for jj in range(score.shape[0]):
                if score[jj] <= tao:
                    pred.append(0)
                else:
                    pred.append(1)
            
            # tn, fp, fn, tp= confusion_matrix(label, pred).ravel()
            # acc = 100*((tp+tn)/(tp+tn+fp+fn))
            
            acc = (score.shape[0] - np.sum(pred)) / (score.shape[0])
            
                
            
            CNN.append(acc)
            
        df3=pd.DataFrame(CNN)
        df3.columns=['acc']
        # if self.epoch== self.opt.niter-1:
        #     df3.to_csv('experiments\{:s}\{:s}\performance_train\{:s}_{:d}_{:s}_per_{:s}.csv'.format(self.opt.dataset,self.opt.tun_name,self.phase,self.epoch,self.opt.tun_name,exp), index=False)
        
        
        plt.plot(thres,CNN, label='Accuracy') 
        plt.title('%s'%exp)
        plt.legend()  
           
        # plt.savefig('experiments\{:s}\{:s}\performance_train\{:s}_{:d}_{:s}_per_img_{:s}.png'.format(self.opt.dataset,self.opt.tun_name,self.phase, self.epoch,self.opt.tun_name,exp))
        plt.close()

    

    
    def performance(self):
        LS1 = self.Tscores[0,:,:]
        LS2 = self.Tscores[1,:,:]
        FS1 = self.Tscores[2,:,:]
        FS2 = self.Tscores[3,:,:]
        
        label = np.zeros((self.Tscores.shape[1],1))
        
        weight = self.opt.per_weight
        
        scores_LS1 = np.sum(weight*np.array(LS1),axis=1)
        scores_LS2 = np.sum(weight*np.array(LS2), axis=1)
        scores_FS1 = np.sum(weight*np.array(FS1),axis=1)
        scores_FS2 = np.sum(weight*np.array(FS2), axis=1)
        
        scores_LS1_norm = (scores_LS1-np.min(scores_LS1))/(np.max(scores_LS1)-np.min(scores_LS1))
        scores_LS2_norm = (scores_LS2-np.min(scores_LS2))/(np.max(scores_LS2)-np.min(scores_LS2))
        scores_FS1_norm = (scores_FS1-np.min(scores_FS1))/(np.max(scores_FS1)-np.min(scores_FS1))
        scores_FS2_norm = (scores_FS2-np.min(scores_FS2))/(np.max(scores_FS2)-np.min(scores_FS2))
        
        
        thres = self.opt.threshold
        ##LS1
        self.per(thres,scores_LS1_norm, label,'LS1')
        
        
        ##LS2
        self.per(thres,scores_LS2_norm, label,'LS2')
        
        
        ##FS1
        self.per(thres,scores_FS1_norm, label,'FS1')
        
        
        ##FS2
        self.per(thres,scores_FS2_norm, label,'FS2')
        
      
