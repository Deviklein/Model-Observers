#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:39:33 2020

@author: Devi

Filters for Channelized Hotelling Observer (CHO)
"""

from baseModel import baseModel
from scipy.special import comb, factorial
import numpy as np 
import matplotlib.pyplot as plt

class ChannelFilters(baseModel):
    def __init__(self, dimensions, radius):
        '''
        Initialize the channel filters for the CHO model.

        Parameters
        ----------
        dimensions : TYPE, tuple of length 2.
            DESCRIPTION. Dimensions of the image (rows,columns)
        radius : TYPE, float.
            DESCRIPTION. radius of the signal template 

        Returns
        -------
        None.

        '''
        super().__init__(dimensions,radius)
    
    def normChannels(self, channel):
        return (channel/np.sum(channel))
    
    def fDoG(self, n, q = 1.67, alpha = 1.4, s0 = .005, norm = True):
        '''
        Create DoG filter in the frequency domain??

        Parameters
        ----------
        n : TYPE, integer.
            DESCRIPTION. 
        q : TYPE, optional
            DESCRIPTION. The default is 1.67.
        alpha : TYPE, optional
            DESCRIPTION. The default is 1.4.
        s0 : TYPE, optional
            DESCRIPTION. The default is .005.

        Returns
        -------
        DoG filter in the frequency domain with the DC component in the 
        center and a dictionary of the parameters used to create the DoG.
        
        '''
        freqs = super().distance(*super().freqDom())
        sj = s0*(alpha**n)
        g1 = np.exp((-1/2)*(freqs/(q*sj))**2)
        g2 = np.exp((-1/2)*(freqs/sj)**2)

        params = {'n': n,
                  'q' : q,
                  'alpha': alpha,
                  's0': s0,
                  }
        dog = g1 -g2
        if norm is True:
            dog = self.normChannels(dog)
        return(dog,params)
     
    def sLGauss(self,a = 8,b= 8, n = 6, norm = True):
        '''
        Generate a Laguerre—Gauss Filter in the spatial domain. 

        Parameters
        ----------
        a : TYPE, optional
            DESCRIPTION. The default is 8.
        b : TYPE, optional
            DESCRIPTION. The default is 8.
        n : TYPE, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        TYPE, 2D numpy array
            DESCRIPTION. The filter in the spatial domain.  

        '''
        xc, yc = 0, 0
        X, Y = super().coordSpace()
        def lgpoly(x):
            total = 0
            for m in range(n):
                current = (-1)**m*comb(n,m)*(x**m/factorial(m))
                total += current
            return(total)
        
        def gamma(x,y):
            return ((2*np.pi)*(((x - xc)**2 / a**2) + ((y - yc)**2 / b**2)))
        
        vfunc = np.vectorize(lgpoly) #vectorize the lgpoly function
        
        g = gamma(X,Y)
        lgp = vfunc(g)
        Clg = np.exp(-.5*g)*lgp
        
        if norm is True:
            Clg = self.normChannels(Clg)
            
        params = {'a': a,
                  'b': b,
                  'n': n}
        return(Clg, params)
    
    def sGab(self, b = 2.5, theta = np.pi/2, Lambda = 20, phi = np.pi/4,
             gamma = 1, k = np.pi, norm = True):
        '''
        Create Gabor filter in spatial domain. 
        
        Code copied from: http://www.cs.rug.nl/~imaging/simplecell.html
            
        Parameters:
            b (type- float): specifies the spatial-frequency bandwidth of the 
                filter. The bandwidth is specified in octaves.
            
            theta (type- float): orientation of the normal to the parallel 
                stripes of a Gabor function. Specified in radians. 
                           
            Lambda (type- int): Wavelength of the sinusoidal (cosine) factor. 
                Specified in pixels. 
            
            phi (type- float): phase offset of cosine factor. Specified in 
                radians. 
            
            gamma (type- float):  spatial aspect ratio, specifies the 
                ellipticity of the support of the Gabor function. Specified as
                a ratio. 
        
        Return:
            
            Gabor filter of size "dimensions" defined in class __init__.
        '''
        sigma = Lambda * ((1/k) * np.sqrt(np.log(2)/2) * (2**b + 1)/(2**b - 1))
        
        params = {'b': b,
                  'theta': theta,
                  'Lambda': round(Lambda,5),
                  'phi': phi,
                  'gamma': gamma,
                  'sigma': sigma,
                  }
        
        if (sigma > self.dim[0]/4) or (sigma > self.dim[1]/4):
            return(None, params) #only return gabors that fit in the image 
                                 #i.e. 2 standard deviations in each direction
        
        # Bounding Box
        x, y = super().coordSpace()

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        gaus = np.exp((-1/(2*sigma**2)) * (x_theta**2 + y_theta**2 * gamma**2))
        sin = np.cos((2 * k) * (x_theta / Lambda) + phi)
        
        gabor = gaus * sin
        
        if norm is True:
            gabor = self.normChannels(gabor)
            
        return(gabor, params)
    
    def channels(self, ftype = 'fDoG', **kwargs):
        '''
        Create a filter bank for a specific filter type. 

        Parameters
        ----------
        ftype : TYPE, optional
            DESCRIPTION. The default is 'fDoG'.
        **kwargs : TYPE, dictionary
            DESCRIPTION. the dictionary maps all parameter value names for the 
            corresponding filter to sequences of those parameter values. This 
            creates a filter bank for the specific filter identified in ftype
            arguement above. 

        Returns
        -------
        Dictionary. mapping 

        '''
        chnls = {}
        if ftype =='fDoG':
            for n in range(1,kwargs['N']+1):
                dogF, params = self.fDoG(n)
                shift = np.fft.ifftshift(dogF)
                sDoG = np.fft.fftshift((np.fft.ifft2(shift).real))
                chnls[n] = (sDoG, params)
        
        elif ftype == 'sLGauss':
            i = 1
            for a_ in kwargs['A']:
                for b_ in kwargs['B']:
                    for n_ in range(1,kwargs['N']+1):
                      chnls[i] = self.sLGauss(a = a_, b= b_, n = n_)
                      i += 1
                      
        elif ftype == 'sGabor':
            i = 0
            for b_ in kwargs['b']:
                for t in kwargs['Theta']:
                    for l in kwargs["Lambda"]:
                        for p in kwargs["Phi"]:
                            for g in kwargs["Gamma"]:
                                chnls[i] = self.sGab(b = b_, theta = t,
                                                     Lambda = l, phi = p,
                                                     gamma = g, norm = True)
                                i += 1
                                
        else:
            er = 'This filter type, {0} has not been implemented here.'
            raise ValueError(er.format(ftype))
        
        return(chnls)
  
    
  
    
  
    
  
    
  
    
###############################################################################
#Main code below
###############################################################################    
if __name__ == "__main__":  
    ch = ChannelFilters((1024,1024),6)
    
    def tempEcc(ecci, a = 0.7063, b =  1.6953,):
        c = (1 + a * (ecci)**b)
        #convert wavelength of sinusoid factor to cycles per degree and scale by c
        lam = [c * 45 / (2**i) for i in range(5)]
        kwargs = {
             'b': [1],
             'Theta': [(i/8)*np.pi for i in range(0,8)],
             'Lambda': lam,
             'Phi': [0],
             'Gamma': [1],
            }
        return kwargs
    
    for p in range(0,11):
        ecc = p
        kwargs = tempEcc(ecc)
        gabors = ch.channels(ftype = 'sGabor', **kwargs)

        fig, axes = plt.subplots(nrows = 8, ncols = 5, figsize = (30,30))
        for i, ax in enumerate(axes.flat):
            gabor = gabors[i][0]
            if gabor is None:
                continue
            params = gabors[i][1]
            
            im = ax.imshow(gabor, cmap = 'gray')
            ax.set_title('Gabor Filter', fontsize = 'large', fontweight = 'bold')
            ax.set_xlabel("Wavelength {0} cpd, sigma {1} pxls".format(np.round(params['Lambda']),np.round(params['sigma'])))
            plt.colorbar(im, ax = ax)
            #plt.tight_layout()
            plt.title("Eccentricity {0}".format(ecc))
            plt.savefig('/Users/Devi/Desktop/Gabor Filters ecc {0}.jpg'.format(ecc))
    
# =============================================================================
#     kwargs = {'A': [5,14,8],
#               'B': [5,14,8],
#               'N': 6,
#               }
#     lgauss = ch.channels(ftype = 'sLGauss', **kwargs)
# 
#     fig, axes = plt.subplots(nrows = 6, ncols = 3, figsize = (20,20))
#     for i, ax in enumerate(axes.flat):
#         i += 1
#         lg = lgauss[i][0]
#         params = lgauss[i][1]
#         
#         im = ax.imshow(lg, cmap = 'gray')
#         ax.set_title('Laguerre—Gauss Filter', fontsize = 'large', fontweight = 'bold')
#         ax.set_xlabel(str(params))
#         plt.colorbar(im, ax = ax)
#         plt.tight_layout()
#         plt.savefig('/Users/Devi/Desktop/Laguerre—Gauss Filters.jpg')
# =============================================================================
        
    
# =============================================================================
#     kwargs = {
#               'N': 10,
#               }
#     dogs = ch.channels(**kwargs)
# 
#     fig, axes = plt.subplots(nrows = 5, ncols = 2, figsize = (20,20))
#     for i, ax in enumerate(axes.flat):
#         i += 1
#         dog = dogs[i][0]
#         params = dogs[i][1]
#         
#         im = ax.imshow(dog, cmap = 'gray')
#         ax.set_title('DoG Filter', fontsize = 'large', fontweight = 'bold')
#         ax.set_xlabel(str(params))
#         plt.colorbar(im, ax = ax)
#         plt.tight_layout()
#         plt.savefig('/Users/Devi/Desktop/DoG Filters.jpg')
# =============================================================================



