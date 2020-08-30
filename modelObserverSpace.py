#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:15:53 2020

@author: Devi
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio
import os

class space:
    def __init__(self, dimensions,):
        '''
        Initialize class. 

        Parameters
        ----------
        dimensions : TYPE, tuple containing 2 (or 3) integers, 
        i.e. len(dimensions) == 2 (or 3).
            DESCRIPTION. (# rows, # columns, optional: # of slices). This 
                          defines the space in which the model observer class
                          will operate in. 

        Returns
        -------
        None.

        '''
        self.dim = dimensions
    
    def __repr__(self,):
        '''
        This method is primaraly used for debugging and for the devloper to 
        see what parameters for the class were fed in to create an instance of 
        this class.

        Returns
        -------
        A string. that represents the syntax used to create this class object.

        '''
        return(f'{self.__class__.__name__}('
               f'{self.dim!r},)')
    
    def twoD_Euclid(self,):
        halfc = math.ceil((self.dim[1] - 1)/2)
        halfr = math.ceil((self.dim[0]- 1)/2)
        cols = np.linspace(-halfc, halfc - 1, self.dim[1])
        rows = np.linspace(halfr, -1*(halfr - 1), self.dim[0])
        x, y = np.meshgrid(cols,rows)
        return(x,y)
    
    def threeD_Euclid(self,):
        halfc = math.ceil((self.dim[1] - 1)/2)
        halfr = math.ceil((self.dim[0]- 1)/2)
        halfs = math.ceil((self.dim[2] -1)/2)
        cols = np.linspace(halfc, -1*(halfc - 1), self.dim[1])
        rows = np.linspace(halfr, -1*(halfr - 1), self.dim[0])
        slices = np.linspace(halfs, -1*(halfs - 1), self.dim[2])
        x, y, z = np.meshgrid(cols, rows, slices)
        return(x, y, z)
    
    def twoD_dist(self,):
        x, y = self.twoD_Euclid()
        return(np.sqrt(x**2 + y**2))
    
    def threeD_dist(self,):
        x, y, z = self.threeD_Euclid()
        return(np.sqrt(x**2 + y**2 + z**2))
    
    def twoD_Freq(self,):
        x, y = self.twoD_Euclid()
        u, v = x/self.dim[1], y/self.dim[0]
        return (u,v)
    
    def threeD_Freq(self,):
        x, y, z = self.threeD_Euclid()
        u, v, s = x/self.dim[1], y/self.dim[0], z/self.dim[2]
        return (u, v, s)
    
    def twoD_radialFreq(self,):
        u,v = self.twoD_Freq()
        return(np.sqrt(u**2 + v**2))
    
    def threeD_radialFreq(self,):
        u, v, s = self.threeD_Freq()
        return(np.sqrt(u**2 + v**2 + s**2))

class nps(space):
    def __init__(self, dimensions, powerLaw):
        super().__init__(dimensions)
        self.p = powerLaw
    
    def normalizeFilter(self,filter_):
        unnorm_filter = np.sqrt(filter_) #unnormailzed filter for NPS
        m_unnorm = unnorm_filter.mean()  #mean of template
        num_pix = 0
        for pix in filter_.shape:
            if num_pix == 0:
                num_pix += pix
            else:
                num_pix *= pix
        var_filter = np.sum(np.abs(unnorm_filter - m_unnorm)**2) / num_pix
        if var_filter == 0:
            return(unnorm_filter)
        else:
            norm_filter = unnorm_filter / np.sqrt(var_filter) #normalize filter 
            return(norm_filter)
        
    def nps2D(self,):
        dist = np.fft.ifftshift(super().twoD_radialFreq()) #shift the DC component to top left corner
        with np.errstate(divide='ignore'): #prevent warning for dividing by zero
            temp = 1/(dist**self.p) #power law filter the spatial frequencies 
        temp[0,0] = temp[0,1] #replace the dc component (inf) with neighboring value
        return(self.normalizeFilter(temp))
    
    def nps3D(self,):
        dist = np.fft.ifftshift(super().threeD_radialFreq())
        with np.errstate(divide='ignore'): #prevent warning for dividing by zero
            temp = 1/(dist**self.p) #power law filter the spatial frequencies 
        temp[0,0,0] = temp[0,1,0] #replace the dc component (inf) with neighboring value  
        return(self.normalizeFilter(temp))

    def loadnps(self, fp, twoD = False):
        nps = sio.loadmat(fp)['S']
        if twoD:
            return (nps.mean(axis = 2))
        else:
            return (nps['S'])
            

class background(nps):
    def __init__(self, dimensions, powerlaw, mu, sigma):
        super().__init__(dimensions, powerlaw)
        self.mu = mu
        self.sigma = sigma 
        self.rc = int(dimensions[0]/2)
        self.cc = int(dimensions[1]/2)
        if len(dimensions) == 3:
            self.nps = self.sigma * super().nps3D()
            self.zc = int(dimensions[2]/2)
            
        elif len(dimensions) == 2:
            self.nps = self.sigma * super().nps2D()
        else:
            raise ValueError('Only 2D or 3D images are accepted.') 
    
    def noise2D(self,):
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
        IID_noise = np.random.normal(0, 1, self.dim)
        Fnoise = np.fft.fft2(IID_noise) #noise in fourier domain
        filterNoise = np.multiply(Fnoise, self.nps)
        spatialNoise = np.fft.ifft2(filterNoise).real + self.mu
        return(spatialNoise)
        
    def noise3D(self,):
        '''
        Generate a background image of uniform gray levels with noise added on
        top of it. The noise can be white noise or filtered depending on the
        value of self.p.

        Returns
        -------
        TYPE, 3D Numpy array
            DESCRIPTION. Correlated or white noise added to a uniform 
            background template.
            
        '''
        IID_noise = np.random.normal(0, 1, self.dim)
        Fnoise = np.fft.fftn(IID_noise) #noise in fourier domain
        filterNoise = np.multiply(Fnoise, self.nps)
        spatialNoise = np.fft.ifftn(filterNoise).real + self.mu
        return(spatialNoise)
    
    def loadNoise(self, filepath):
        return(sio.loadmat(filepath))
        pass

class signal(space):
    def __init__(self, dimensions, signalRadius = 3, signalContrast = 1,
                 sigma = 10):
        super().__init__(dimensions)
        self.sr = signalRadius
        self.sc = signalContrast
        self.sig = sigma
    
    def signalSpace(self):
        emptyTemplate = np.zeros(self.dim)
        if len (self.dim) == 2:
            distanceField = super().twoD_dist()
        elif len(self.dim) == 3:
            distanceField = super().threeD_dist()
            
        return(emptyTemplate, distanceField)
    
    def calc(self,):
        signal_template, dist = self.signalSpace()
        condition = dist <= self.sr
        signal_template[condition] += self.sc
        return (signal_template)
    
    def mass(self,):
        signal_template, dist = self.signalSpace()
        temp_template = np.exp(-(dist**2/(2*self.sig**2))**len(self.dim))
        condition = temp_template > 1e-7 
        signal_template[condition] += temp_template[condition]
        return(self.sc * signal_template)
    
class trial:
    def __init__(self, dimensions,):
        assert (len(dimensions) == 2 ) or (len(dimensions) == 3) #check for
                                                                 #user bugs
        self.dim = dimensions
        self.rc = int(dimensions[0]/2)
        self.cc = int(dimensions[1]/2)
        if len(dimensions) == 3:
            self.sc = int(dimensions[2]/2)
        
    def addSignal2D(self, im, sig, row, col,):
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

        max_r = self.dim[0] - 1
        max_c = self.dim[1] - 1
        t, t_ = row - self.rc, 0
        b, b_ = row + self.rc, self.dim[0]
        l, l_ = col - self.cc, 0
        r, r_ = col + self.cc, self.dim[1]
        
        if l < 0:
            l = 0
            l_ = self.cc - col
        if r > max_c:
            r = max_c
            r_ = max_c - col + self.cc 
        if t < 0:
            t = 0
            t_ = self.rc - row
        if b > max_r:
            b = max_r
            b_ = max_r - row + self.rc
                
        im[t:b, l:r] += sig[t_:b_, l_:r_]
        
        return(im,(row,col))
    
    def addSignal3D(self, im, sig, row, col, slc):
        '''
        Add the signal profile to a background template with additive noise.

        Parameters
        ----------
        row : TYPE, int
            DESCRIPTION. The row index at which the center of the signal 
            appears.
        col : TYPE, int
            DESCRIPTION. The column index at which the center of the signal 
            appears.
        slc : Type, int
            DESCRIPTION. The slice (z) index at which the center of the signal 
            appears.

        Returns
        -------
        TYPE, 3D Numpy array
            DESCRIPTION. This is a signal present trial, i.e. additive signal
            embedded in filtered white noise. 
        
        '''

        max_r = self.dim[0] - 1
        max_c = self.dim[1] - 1
        max_s = self.dim[2] - 1
        t, t_ = row - self.rc, 0
        b, b_ = row + self.rc, self.dim[0]
        l, l_ = col - self.cc, 0
        r, r_ = col + self.cc, self.dim[1]
        u, u_ = slc - self.sc, 0
        d, d_ = slc + self.sc, self.dim[2]
        
        if l < 0:
            l = 0
            l_ = self.cc - col
        if r > max_c:
            r = max_c
            r_ = max_c - col + self.cc 
        if t < 0:
            t = 0
            t_ = self.rc - row
        if b > max_r:
            b = max_r
            b_ = max_r - row + self.rc
        if u < 0:
            u = 0
            u_ = self.sc - slc
        if d > max_s:
            d = max_s
            d_ = max_s - slc + self.sc
                
        im[t:b, l:r, u:d] += sig[t_:b_, l_:r_, u_:d_]
        return(im,(row,col,slc))
    
if __name__ == "__main__":  
    #Example code
    
    #2D example
    dim = (128,128)
    f = 2.8
    mu = 128
    sigma = 30
    contrast = 83
    sigRadius = 3

    location = (int(dim[0]/2),int(dim[1]/2))
    
    
    bckg = background(dim, f, mu, sigma)
    sig = signal(dim, signalRadius = sigRadius, signalContrast = contrast)
    signal = sig.calc()
    tr = trial(dim)
    sigPres = tr.addSignal2D(bckg.noise2D(), signal, *location)
    
    
    #Plot the trial
    plt.figure()
    plt.title("Signal Present Trial")
    plt.imshow(sigPres[0][:,:], cmap = 'gray')
    
    ###NPW model observer
    trial_f = np.fft.fft2(sigPres[0][:,:])
    signal_f = np.fft.fft2(np.fft.fftshift(signal))
    npw = np.fft.ifft2(trial_f * signal_f).real
    plt.figure()
    plt.title("NPW template")
    plt.imshow(npw[:,:],cmap = 'gray')
    
    
    ###Ideal Observer 
    w_f = np.divide(signal_f,bckg.nps2D()**2)
    io = np.fft.ifft2(w_f * trial_f).real
    plt.figure()
    plt.title("IOtemplate")
    plt.imshow(io[:,:], cmap = 'gray')
    
    #3D example
# =============================================================================
#     dim = (128,128,10)
#     f = 2.8
#     mu = 128
#     sigma = 30
#     contrast = 150
#     sigRadius = 3
# 
#     location = (int(dim[0]/2),int(dim[1]/2),int(dim[2]/2))
#     
#     
#     bckg = background(dim, f, mu, sigma)
#     sig = signal(dim, signalRadius = sigRadius, signalContrast = contrast)
#     signal = sig.calc()
#     tr = trial(dim)
#     sigPres = tr.addSignal3D(bckg.noise3D(), signal, *location)
#     
#     
#     ###NPW model observer
#     plt.figure()
#     plt.imshow(sigPres[0][:,:,int(dim[2]/2)], cmap = 'gray')
#     trial_f = np.fft.fftn(sigPres[0][:,:,:])
#     signal_f = np.fft.fftn(np.fft.fftshift(signal))
#     npw = np.fft.ifftn(trial_f * signal_f).real
#     plt.figure()
#     plt.title("NPW template")
#     plt.imshow(npw[:,:,int(dim[2]/2)],cmap = 'gray')
#     
#     
#     ###Ideal Observer 
#     w_f = np.divide(signal_f,bckg.nps3D()**2)
#     io = np.fft.ifftn(w_f * trial_f).real
#     plt.figure()
#     plt.imshow(io[:,:,int(dim[2]/2)], cmap = 'gray')
# =============================================================================


# =============================================================================
# if len(dim) == 2:
#     plt.close('all')
#     plt.figure()
#     plt.imshow(sigPres[0], cmap = 'gray')
# 
# else:
#     for i in range(location[2]-10, location[2] + 10):
#         plt.figure()
#         plt.imshow(sigPres[0][:,:,i],cmap = "gray")
# 
# =============================================================================
