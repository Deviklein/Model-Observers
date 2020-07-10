#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:06:22 2020

@author: Devi

NPW (Non Prewhiting) Model Observer Template. 
"""

from baseModel import baseModel
import matplotlib.pyplot as plt
import numpy as np

class NPW(baseModel):    
    def __init__(self, d, r):
        '''
        Magic Method.

        Parameters
        ----------
        d : TYPE, tuple
            DESCRIPTION. Dimensions of the template. (# rows, # columns).
        r : TYPE, int.
            DESCRIPTION. The radius of the disk, which is the signal in this
            model observer class for now.

        Returns
        -------
        None.
        
        '''
        super().__init__(d,r)
   
    def template(self, fourier = False):
        '''
        Generate the NPW template.

        Parameters
        ----------
        fourier : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE, 2D numpy array
            DESCRIPTION. The NPW template in the spatial or fourier domain. s

        '''
        sig = super().diskSignal(normalize = False)
        if fourier is False:
            return(sig)
        else:
            return(np.fft.fft2(np.fft.fftshift(sig)))










###############################################################################
#Main code below
###############################################################################       
#Example code to create NPW template
if __name__ == "__main__":
    ###Parameters
    params = [
              (100,100), #figure size 
              6,       #Signal radius in pixel units
              ]
    model = NPW(*params)
    f = model.template()
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
    im = ax.imshow(f, cmap = 'gray')
    ax.set_title('NPW Template', fontsize = 'large', fontweight = 'bold')
    plt.colorbar(im)
    #plt.savefig('/Users/Devi/Desktop/NPW_Template.jpg')