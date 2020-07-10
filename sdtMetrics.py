#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:26:40 2019

@author: Devi

Signal detection theory Analysis class for simple yes/no discrimination tasks.
This class is used to get sensitivity measures (e.g. d' and AUC) and plot ROC 
curves.
"""

import numpy as np 
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, ground_truth, responses):
        '''
        Magic method.

        Parameters
        ----------
        ground_truth : TYPE, sequence.
            DESCRIPTION. A sequence of 1s and 0s. These values correspond to 
            the ground truth for the trial with the corresponding index in the
            "responses" parameter. A 1 represents a signal present trial and 
            a 0 represents a signal absent trial. 
        responses : TYPE, sequence.
            DESCRIPTION. A sequence of the same length as "ground_truth",
            where each element corresponds to a response of a human or model
            observer.

        Returns
        -------
        None.

        '''
        self.gt = np.array(ground_truth)
        self.res = np.array(responses)
         
    def criterion(self, num_criterion):
        '''
        A sequence of criterion values of equal spacing that span the range 
        of the criterion values (i.e., start at the smallest criterion value
        and end at the largest criterion value). 

        Parameters
        ----------
        num_criterion : TYPE, int
            DESCRIPTION. The number of criterions to produce.

        Returns
        -------
        TYPE, float
            DESCRIPTION. A sequence of criterion values of equal spacing. 
        
        '''
        array = self.res
        max_ = np.amax(array)
        min_ = np.amin(array)
        return(np.linspace(min_,max_, num = num_criterion))
    
    def hr_fa(self, criterion):
        '''
        Compute a (H,F) pair for a given criterion. 

        Parameters
        ----------
        criterion : TYPE, float
            DESCRIPTION. Threshold used to compute hits and false alarms.

        Returns
        -------
        TYPE, float.
            DESCRIPTION. A hit rate given the parameter "criterion."
        TYPE, float. 
            DESCRIPTION. A false alarm rate given the parameter "criterion."
        
        '''
        resp_yes = self.gt[self.res > criterion]
        
        hits = np.sum(resp_yes)
        sigpres = np.sum(self.gt)
        
        fa = resp_yes.shape[0] - hits 
        sigabsent = self.gt.shape[0] - sigpres
        #print(sigabsent, sigpres)
        return (hits/sigpres, fa/sigabsent)
    
    def integrate(self,x ,y):
        '''
        Compute the empirical area under the ROC curve (eAUC). 

        Parameters
        ----------
        x : TYPE, array.
            DESCRIPTION. A list of false alarm rates.
        y : TYPE, array.
            DESCRIPTION. A list of true positive rates.

        Returns
        -------
        TYPE, float
            DESCRIPTION. The empirical area under the curve. 
        
        '''
        def moving_average(x, w):
            #convolution helper function
            return np.convolve(x, np.ones(w), 'valid') / w
        
        diff = np.diff(np.array(x))
        avg = moving_average(np.array(y),2)
        
        return abs(np.multiply(diff, avg).sum())
    
    def ROC(self, num_crit = 1000, plot = True):
        '''
        Plot ROC curve and compute AUC (empirically, i.e. compute eAUC).

        Parameters
        ----------
        num_crit : TYPE, int
            DESCRIPTION. The number of criteria to use to generate the ROC
            curve. The larger the number of criteria, the smoother the curve
            and the better estimate of AUC.
        plot : TYPE, optional
            DESCRIPTION. The default is True. Plot the ROC curve 

        Returns
        -------
        TYPE, float.
            DESCRIPTION. The empirical area under the ROC curve.
        TYPE, array.
            DESCRIPTION. A sequence of false alarm rates used to produce
            the ROC plot in this function and to compute the eAUC value. 
        TYPE, array.
            DESCRIPTION.  A sequence of hit rates used to produce the ROC plot
            in this function and to compute the eAUC value.
       
        '''
        crit_arr = self.criterion(num_crit)
        FA_arr = []
        hit_arr = []
        for crit in crit_arr:
            performance = self.hr_fa(crit)
            hit_arr.append(performance[0])
            FA_arr.append(performance[1])
            
        eauc = self.integrate(FA_arr, hit_arr)
        
        if plot == True:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize= (4,4))
           # ax=fig.add_subplot(1,3,3)
            lab = "IO w/eAUC: {0}".format(round(eauc,3))
            ax.plot(FA_arr,hit_arr, label = lab)
            ax.plot([0,1],[0,1], label = "chance performance w/auc: 0.5")
            ax.set_title('ROC Curve')
            ax.set_xlabel('False Alarm Rate')
            ax.set_ylabel('Hit Rate')
            ax.axis('square')
            ax.axis([-0.01,1.01,-0.01,1.01])
            ax.legend()
            plt.tight_layout()
            
        return(eauc, FA_arr, hit_arr)









###############################################################################
#Main code below
###############################################################################
#if __name__ == "__main__":
