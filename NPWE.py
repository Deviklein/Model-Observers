#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:08:20 2020

@author: Devi

NPWE (Non PreWhitining Eye Filter) Model Observer template.
"""
  
from baseModel import baseModel
import numpy as np
import matplotlib.pyplot as plt

class NPWE(baseModel):
    def __init__(self, dimensions, r):
        '''
        Magic Method.

        Parameters
        ----------
        dimensions : TYPE, tuple
            DESCRIPTION. Dimensions of the template. (# rows, # columns)
        r : TYPE, int.
            DESCRIPTION. The radius of the disk, which is the signal in this
            model observer class for now.

        Returns
        -------
        None.
        
        '''
        super().__init__(dimensions,r)
        self.diskSignal = super().diskSignal()
    
    def eyeFilter(self, ppd = 45, n = 1.3, c = .04, y = 2):
        '''
        Generate the eye filter in the frequency domain. 

        Parameters
        ----------
        ppd : TYPE, optional
            DESCRIPTION. The default is 45. Pixels per degree of visual angel.
            In order to compute this the observer needs to know the viewing 
            distance of the obser to the monitor and the monitor resolution.
        n : TYPE, optional
            DESCRIPTION. The default is 1.3. This is a parameter used in the 
            eye filter template.
        c : TYPE, optional
            DESCRIPTION. The default is .04. This is another parameter used in 
            the eye filter template.
        y : TYPE, optional
            DESCRIPTION. The default is 2. This is another parameter used in 
            the eye filter template.

        Returns
        -------
        TYPE, 2D numpy array.
            DESCRIPTION. The eye filter in the frequency domain. 
        TYPE, dictionary.
            DESCRIPTION. Parameters used to create the eye filter in the 
            frequency domain. 
        
        '''
        freqs = super().freqDom() #units cycles per pixel
        u,v = freqs[0] * ppd, freqs[1] * ppd #units cycles per degree
        rf = super().distance(u, v) 
        params = {'ppd':ppd,
                  'n': n,
                  'c': c,
                  'y': y,}
        return(rf**n*np.exp((-c)*(rf**y)), params)
    
    def eyeTemplate(self, fourier = False):
        '''
        Create the template for the NPWE model observer.

        Parameters
        ----------
        fourier : TYPE, optional
            DESCRIPTION. The default is False. If True, return the template in
            the frequency domaain, otherwise return it in the spatial domain. 

        Returns
        -------
        TYPE, 2D array.
            DESCRIPTION. Model observer template in the frequency or spatial 
            domain.
        TYPE, dictionary.
            DESCRIPTION. Parameters used to create the eye filter in the 
            frequency domain. 

        '''
        sF = np.fft.fft2(self.diskSignal)
        eyeF, params = self.eyeFilter()
        eyeF = np.fft.ifftshift(eyeF)
        sFeF = np.multiply(sF, eyeF)
        if fourier is False:
            return(np.fft.ifft2(sFeF).real, params)
        else:
            return(sFeF, params)
  
    
  
    
  
    
  
    
  
    
###############################################################################
#Main code below
###############################################################################
if __name__ == "__main__":
    params = [(100,100),
              6,
              ]
    npwe = NPWE(*params)
    f, parameters = npwe.eyeTemplate()
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
    im = ax.imshow(f, cmap = 'gray')
    ax.set_title('NPWE Template', fontsize = 'large', fontweight = 'bold')
    plt.colorbar(im)
    #plt.savefig('/Users/Devi/Desktop/NPWE_Template.jpg')