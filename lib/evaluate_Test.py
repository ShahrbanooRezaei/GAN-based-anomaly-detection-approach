""" 
Developed by Shahrbanoo Rezaei 1/1/2021

Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
    Sensitivity, Specificity, Accuracy, F1, PPV
"""

import numpy as np
import pandas as pd
from numpy import genfromtxt
from options import Options
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve


opt = Options().parse()

def per(thres,score, label,exp, epoch):
    """ Compute Accuracy, sensitivity, specificity, F1 score and PPV for different thresholds """
    CNN = [] 
    pred = []
    acc=[] 
    sensetivity=[] 
    specificity=[] 
    F1=[] 
    ppv=[]
    
    for tao in thres:
        pred = []
        for jj in range(score.shape[0]):
            if score[jj] <= tao:
                pred.append(0)
            else:
                pred.append(1)
        
        tn, fp, fn, tp= confusion_matrix(label, pred).ravel()
        if tao==0.3:
            print(exp,tao,tn, fp, fn, tp)
        acc = 100*((tp+tn)/(tp+tn+fp+fn))
        sensetivity = 100*( tp/(tp + fn))
        specificity = 100*( tn/(tn + fp))
        F1 = 100*((2*tp) / (2*tp + fp + fn))
        ppv = 100*( tp/(tp+fp+0.000005))
            
        
        CNN.append([tao,acc, sensetivity, specificity, F1, ppv])
        
    df3=pd.DataFrame(CNN)
    df3.columns=['Tao', 'acc', 'sensetivity', 'specificity', 'F1', 'PPV']
    df3.to_csv('experiments\{:s}\{:s}\performance\{:d}_fper_{:s}.csv'.format(opt.dataset,opt.tun_name,epoch,exp), index=False)
    
    fig = plt.figure(figsize=(8,8))
    j=0
    C = np.array(CNN)
    # plt.title('%s'%exp)
    for t in ['Accuracy', 'sensetivity', 'specificity', 'F1', 'PPV']:
        j=j+1
        # plt.subplot(3, 2, j)
        # plt.plot(thres,C[:,j], label='%s'%t)
        ax = fig.add_subplot(3, 2, j)
        ax.plot(thres,C[:,j], label='%s-%s'%(t,exp)) 
        plt.legend()  
      
    plt.savefig('experiments\{:s}\{:s}\performance\{:d}_per_img_{:s}.png'.format(opt.dataset,opt.tun_name,epoch,exp))
    plt.close()
    j=0
    for t in ['Accuracy', 'sensetivity', 'specificity', 'F1', 'PPV']:
        j=j+1
        plt.plot(thres,C[:,j],label='%s-%s'%(t,exp))
        plt.legend()
    plt.savefig('experiments\{:s}\{:s}\performance\{:d}_perT_img_{:s}.png'.format(opt.dataset,opt.tun_name,epoch,exp))
    plt.close()

def per_rev(thres,score, label,exp, epoch):
    """ Compute Accuracy, sensitivity, specificity, F1 score and PPV for different thresholds """
    CNN = [] 
    pred = []
    acc=[] 
    sensetivity=[] 
    specificity=[] 
    F1=[] 
    ppv=[]
    
    for tao in [0.04]:
        pred = []
        for jj in range(score.shape[0]):
            if score[jj] <= tao:
                pred.append(0)
            else:
                pred.append(1)
        
        tn, fp, fn, tp= confusion_matrix(label, pred).ravel()
        
        acc = 100*((tp+tn)/(tp+tn+fp+fn))
        sensetivity = 100*( tp/(tp + fn))
        specificity = 100*( tn/(tn + fp))
        F1 = 100*((2*tp) / (2*tp + fp + fn))
        ppv = 100*( tp/(tp+fp+0.000005))
            
        
        CNN.append([tao,acc, sensetivity, specificity, F1, ppv])
        
    df3=pd.DataFrame(CNN)
    df3.columns=['Tao', 'acc', 'sensetivity', 'specificity', 'F1', 'PPV']
    df3.to_csv('experiments\{:s}\{:s}\performance\{:d}_fper_{:s}.csv'.format(opt.dataset,opt.tun_name,epoch,exp), index=False)
    
     
      
    

def roc(scores, labels, exp, epoch):
    """Compute ROC curve and ROC area for each class"""
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
    plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
    plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for %s' %exp)
    plt.legend(loc="lower right")
    plt.savefig('experiments\{:s}\{:s}\performance\{:d}_ROC_img_{:s}.png'.format(opt.dataset,opt.tun_name,epoch,exp))
    #plt.show()
    plt.close()

def PR(scores, labels,exp, epoch):
    """Compute PR curve """
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    PR_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw, label='(AUC = %0.2f)' % (PR_auc))
    plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall for %s' %exp)
    plt.legend(loc="lower right")
    plt.savefig('experiments\{:s}\{:s}\performance\{:d}_PR_img_{:s}.png'.format(opt.dataset,opt.tun_name,epoch,exp))
    #plt.show()
    plt.close()    
    
def performance(opt,epoch):
    
    score_root_1 = 'experiments/'+'%s/'%opt.dataset+'%s/'%opt.tun_name+'performance/'+ 'fe_'+'%d'%epoch +'_Score_FS1.csv'
    score_root_2 = 'experiments/'+'%s/'%opt.dataset+'%s/'%opt.tun_name+'performance/'+ 'fe_'+'%d'%epoch+'_Score_FS2.csv'
    label_root = opt.label_root
    
    FS1 = np.array(pd.read_csv(score_root_1, header=None))
    FS2 = np.array(pd.read_csv(score_root_2, header=None))
    #???????????????????????????????????????????? 1300
    
    weight = opt.per_weight
    
    scores_FS1 = np.sum(weight*(FS1),axis=1)
    scores_FS2 = np.sum(weight*(FS2), axis=1)
    label_FS1_FS2 = np.array(pd.read_csv(label_root, header=None))
    
    
    scores_FS1_norm = (scores_FS1-np.min(scores_FS1))/(np.max(scores_FS1)-np.min(scores_FS1))
    scores_FS2_norm = (scores_FS2-np.min(scores_FS2))/(np.max(scores_FS2)-np.min(scores_FS2))
    
    
    thres = opt.threshold
    ##FS1
    per(thres,scores_FS1_norm, label_FS1_FS2[:,0],'FS1',epoch)
    roc(scores_FS1_norm, label_FS1_FS2[:,0],'FS1',epoch)
    PR(scores_FS1_norm, label_FS1_FS2[:,0],'FS1',epoch)
    
    ##FS2
    per(thres,scores_FS2_norm, label_FS1_FS2[:,0],'FS2',epoch)#!!!!!!!!!!!!!!
    roc(scores_FS2_norm, label_FS1_FS2[:,0],'FS2',epoch)#!!!!!!!!!!!!!!!
    PR(scores_FS2_norm, label_FS1_FS2[:,0],'FS2',epoch)


def performance_rev(opt,ep, epoch):
    
    score_root_1 = 'experiments/'+'%s/'%opt.dataset+'%s/'%opt.tun_name+'performance/'+ 'fe_'+'%d'%ep +'_Score_FS1.csv'
    score_root_2 = 'experiments/'+'%s/'%opt.dataset+'%s/'%opt.tun_name+'performance/'+ 'fe_'+'%d'%ep+'_Score_FS2.csv'
    label_root = opt.label_root
    
    FS1 = np.array(pd.read_csv(score_root_1, header=None))
    FS2 = np.array(pd.read_csv(score_root_2, header=None))
    #???????????????????????????????????????????? 1300
    
    weight = opt.per_weight
    
    scores_FS1 = np.sum(weight*(FS1),axis=1)
    scores_FS2 = np.sum(weight*(FS2), axis=1)
    label_FS1_FS2 = np.array(pd.read_csv(label_root, header=None))
    
    
    scores_FS1_norm = (scores_FS1-np.min(scores_FS1))/(np.max(scores_FS1)-np.min(scores_FS1))
    scores_FS2_norm = (scores_FS2-np.min(scores_FS2))/(np.max(scores_FS2)-np.min(scores_FS2))
    
    
    thres = opt.threshold
    ##FS1
    per_rev(thres,scores_FS1_norm, label_FS1_FS2[:,0],'FS1',epoch)
    # roc(scores_FS1_norm, label_FS1_FS2[:,0],'FS1',epoch)
    # PR(scores_FS1_norm, label_FS1_FS2[:,0],'FS1',epoch)
    
    ##FS2
    per_rev(thres,scores_FS2_norm, label_FS1_FS2[:,0],'FS2',epoch)#!!!!!!!!!!!!!!
    # roc(scores_FS2_norm, label_FS1_FS2[:,0],'FS2',epoch)#!!!!!!!!!!!!!!!
    # PR(scores_FS2_norm, label_FS1_FS2[:,0],'FS2',epoch)
    
 