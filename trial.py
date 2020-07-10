#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:49:27 2020

@author: Devi

This class will simulate images with filtered noise. 

The images generated in this class can be fed into the model observers to run
simulations.

Currently, this class only supports generation of 2D images. 
"""

from baseModel import baseModel
import numpy as np
import matplotlib.pyplot as plt

class trial(baseModel):
    def __init__(self, dim, p, sr, c, mu, sigma,):
        '''
        Magic Method.

        Parameters
        ----------
        dim : TYPE, tuple (#rows, #cols)
            DESCRIPTION. The size of the image.
        p : TYPE, float
            DESCRIPTION. exponent used to filter IID gaussian distributed 
            noise. See baseModel.Filter docstring for specific implementation
            details. 
        sr : TYPE, int
            DESCRIPTION. The radius of the disk signal. As of right now,
            the only signal profile that has been implmented is a circular disk
        c : TYPE, int
            DESCRIPTION. The "contrast" of the signal. In other words, how many
            gray levels should the signal profile be increased above 
            the background gray level value(s).
        mu : TYPE, int
            DESCRIPTION. The mean background level of the image. Typically this
            is set to 128 (gray background given stored as 8 bit pixels)
        sigma : TYPE, int
            DESCRIPTION. The desired standard deviation for the random noise
            matrix. 

        Returns
        -------
        None.
        
        '''
        super().__init__(dim, sr)
        self.d = dim
        self.p = p 
        self.sr = sr
        self.c = c
        self.mu = mu
        self.sigma = sigma
        self.rc = int(dim[0]/2)
        self.cc = int(dim[1]/2)
        self.signal = c * super().diskSignal(normalize = False) 
        self.npsv = sigma * super().NPS(p)
        
    def noiseImg(self):
        '''
        Generate a background image of uniform gray levels with noise added on
        top of it. The noise can be white noise or filtered depending on the
        value of self.p.

        Returns
        -------
        TYPE, 2D Numpy array
            DESCRIPTION. Correlated or white noise added to a uniform 
            background template.
            
        '''
        IID_noise = np.random.normal(0, 1, self.d)
        Fnoise = np.fft.fft2(IID_noise) #noise in fourier domain
        filterNoise = np.multiply(Fnoise, self.npsv)
        spatialNoise = np.fft.ifft2(filterNoise).real + self.mu
        return(spatialNoise)
    
    def diskInBounds(self, row = None, col = None):
        '''
        Determine the appropriate location for the signal profile center 
        (row,  col) indices. 
                                                                         
        Parameters
        ----------
        row : TYPE, optional
            DESCRIPTION. The default is None. enter custom row index, otherwise
            randomly choose a row index in the bounds of the image such that
            the signal is not cut off by the bounds of the image size.
        col : TYPE, optional
            DESCRIPTION. The default is None. enter custom column index, 
            otherwise choose a col index in the... (same as above for rows)

        Returns
        -------
        TYPE, int
            DESCRIPTION. Row index for center of signal profile.
        TYPE, int.
            DESCRIPTION. Column index for center of signal profile
        
        '''
        boundrytop = self.sr 
        boundrybot = self.d[0] - 1 - (self.sr)
        boundryleft = boundrytop
        boundryright = self.d[1] - 1 - (self.sr)
        
        if (row == None) & (col == None):
            row = np.random.randint(boundrytop, boundrybot + 1)
            col = np.random.randint(boundryleft, boundryright + 1)
        elif (row == None) & (col != None):
            assert (col >= boundryleft) & (col <= boundryright)
            row = np.random.randint(boundrytop, boundrybot + 1)
        elif (row != None) & (col == None):
            assert (row >= boundrytop) & (row <= boundrybot)
            col = np.random.randint(boundryleft, boundryright + 1)
        else:
            assert (col >= boundryleft) & (col <= boundryright)
            assert (row >= boundrytop) & (row <= boundrybot)
            
        return(row,col)
    
    def addSignal(self, row = None, col = None, **kwargs):
        '''
        Add the signal profile to a background template with additive noise.

        Parameters
        ----------
        row : TYPE, optional
            DESCRIPTION. The default is None. See row parameter description in 
            "diskInBounds" method.
        col : TYPE, optional
            DESCRIPTION. The default is None. See col parameter description in 
            "diskInBounds" method.

        Returns
        -------
        TYPE, 2D Numpy array
            DESCRIPTION. This is a signal present trial, i.e. additive signal
            embedded in filtered white noise. 
        
        '''
        
        if 'im' not in kwargs.keys():
            im = self.noiseImg()
        else:
            im = kwargs['im']
        
        if 'sig' not in kwargs.keys():
            sig = self.signal
            row, col = self.diskInBounds(row = row, col = col)
        else:
            sig = kwargs['sig']
            
        max_r = self.dim[0] - 1
        max_c = self.dim[1] - 1
        d_r = self.rc
        d_c = self.cc
        t, t_ = row - d_r, 0
        b, b_ = row + d_r, self.dim[0]
        l, l_ = col - d_c, 0
        r, r_ = col + d_c, self.dim[1]
        
        if l < 0:
            l = 0
            l_ = d_c - col
        if r > max_c:
            r = max_c
            r_ = max_c - col + d_c  
        if t < 0:
            t = 0
            t_ = d_r - row
        if b > max_r:
            b = max_r
            b_ = max_r - row + d_r
                
        im[t:b, l:r] += sig[t_:b_, l_:r_]
        return(im,(row,col))
    
    def simTrial(self, LKE = True, signalpresent = True, fourier = False, 
                 **kwargs):
        '''
        Simulate a signal present trial or signal absent trial. 

        Parameters
        ----------
        LKE : TYPE, optional
            DESCRIPTION. The default is True. If "True", signal will be placed 
            at center of the image, signal location known exactly detection
            task. Otherwise it is a search task where the signal will be placed 
            randomly in the image boundaries. 
        signalpresent : TYPE, optional
            DESCRIPTION. The default is True. If "True", simulate a signal 
            present trial. Otherwise simulate a signal absent trial.
        fourier : TYPE, optional
            DESCRIPTION. Whether to return the the trial tempalte in the 
            spatial or fourier domain.
        
        Returns
        -------
        TYPE, 2D numpy array
            DESCRIPTION. Either a signal present trial or a signal absent 
            trial. In either the spatial or the frequency domain. 
        
        '''
        if len(kwargs.keys()) == 0:
            r, c = self.rc, self.cc
        else:
            r, c = kwargs['row'], kwargs['col']
            
        if (LKE is True) & (signalpresent is True) & (fourier is True):
            row, col = r, c
            im, loc = self.addSignal(row, col)
            return(np.fft.fft2(im),loc)
        
        elif (LKE is True) & (signalpresent is True) & (fourier is False):
            row, col = r, c
            im, loc = self.addSignal(row, col)
            return(im,loc)
        
        elif (LKE is False) & (signalpresent is False) & (fourier is False):
            return(self.noiseImg(), None)
            
        elif (LKE is False) & (signalpresent is False) & (fourier is True):
            return(np.fft.fft2(self.noiseImg()), None)
            
        elif (LKE is False) & (signalpresent is True) & (fourier is False):
            im, loc = self.addSignal()
            return(im,loc)
            
        elif (LKE is False) & (signalpresent is True) & (fourier is True):
            im, loc = self.addSignal()
            return(np.fft.fft2(im), loc)
        
        else:
            er = 'No such trial where Location known exactly is True but '
            er1 = 'signal presence is false.'
            raise ValueError(er + er1)                
    
    def sim_trials(self, numPres, numAbs, LKE = True, flatten = False, 
                   fourier = False):
        '''
        create a dictionary that stores a sequence of simulated trials with
        an associated groundtruth sequence. This is basically a wrapper 
        function for "self.simTrial()".

        Parameters
        ----------
        numSigPres : TYPE, int
            DESCRIPTION. Number of signal present trials.
        numSigAbs : TYPE, int
            DESCRIPTION. Number of signal absent trials. 
        LKE : TYPE, optional
            DESCRIPTION. The default is True. Simulate location known excatly
            (LKE) signal present trials if True. Otherwise randomize the 
            location of the signal.
        flatten : TYPE, optional
            DESCRIPTION. The default is False. Should the img be flattened 
            into a vector or left as a 2D array. "flatten" arguement is used
            for model observer classes.
        
        fourier : TYPE, optional
            DESCRIPTION. The default is True. Therefore, return the trial 
            image in the frequency domain. Otherwise, return it in the 
            spatial domain.
        
        Returns
        -------
        TYPE, dictionary
            DESCRIPTION. "ground truth" key maps to a sequence of ground 
            truth values (1s = signal present and 0s = signal absent).
            DESCRIPTION. "images" key maps to a sequence of trial images.
        
        '''
        
        
        ground_truth = numPres * [1] + numAbs * [0] 
        trial_images = []
        
        for tri in ground_truth:
            if tri == 0:
                tri = False
            else:
                tri = True
            t = self.simTrial(LKE = LKE, signalpresent = tri,
                              fourier = fourier)
            
            if flatten == True:
                t = t.flatten().reshape(-1,1)
                
            trial_images.append(t)
                
        return ({'ground truth' : ground_truth, 'images' : trial_images})
  








      
###############################################################################
#Main code below
###############################################################################       
#Example code to create NPW template
if __name__ == "__main__":  
    var_ = []
    t = trial((128,128), 0, 6, 20, 128, 30,)
    tr = t.simTrial(LKE = False, signalpresent = False, fourier = False,)
    
# =============================================================================
#     for i in range(1000):
#         
#         var_.append(t.noiseImg().var(ddof = 1))
#     print(np.array(var_).mean())
# =============================================================================
# =============================================================================
#     a = []
#     for i in range(10000):
#         a.append(t.noiseImg().var())
#     print(np.array(a).mean())
# =============================================================================
# =============================================================================
#     t.simTrial(SKE = True, signalpresent = True)
#     ts = t.sim_trials(1000,1000, SKE = False)
#     plt.imshow(ts['images'][0], cmap = 'gray')
# =============================================================================
    
    