#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:09:35 2020

@author: Devi

HO (Hotelling Observer) Model.
"""
from trial import trial
import numpy as np
import matplotlib.pyplot as plt

class HO(trial):
    '''
    Create a template for the hotelling Observer. 
    '''
    def __init__(self, dim, p, sr, c, mu, sigma,):
        super().__init__(dim, p, sr, c, mu, sigma)
        
    def template(self, fourier = False):
        signal = np.fft.fftshift(super().diskSignal(normalize = True))
        templateF = np.divide(np.fft.fft2(signal), self.npsv)
        if fourier is True:
            return(templateF)
        else:
            templateS = np.fft.ifftshift(np.fft.ifft2(templateF)).real
            return(templateS)










###############################################################################
#Main code below
###############################################################################       
#Example code to create NPW template
if __name__ == "__main__":
    ###Parameters
    params = [
          (100,100), #figure size 
          2.8,       #Power to filter noise usually use 2.8
          6,         #Signal radius in pixel units    
          30,        #signal contrast
          128,       #mean background level
          90,        #variance of gaussian noise
          ]
    model = HO(*params)
    f = model.template(fourier = False)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
    im = ax.imshow(f, cmap = 'gray')
    ax.set_title('HO Template', fontsize = 'large', fontweight = 'bold')
    plt.colorbar(im)
    #plt.savefig('/Users/Devi/Desktop/HO_Template.jpg')

    

        