#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:10:06 2020

@author: Devi

CHO (Channelized Hotelling Observer) Model. 
"""

from ChannelFilters import ChannelFilters
from scipy.stats import norm
from trial import trial

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sci
import pandas as pd

class CHO(ChannelFilters):
    def __init__(self, d, p, sr, c, ppd, mu, sigma, matlab = False):
        super().__init__(d, sr, matlab = matlab)
        self.trial = trial(d, p, sr, c, mu, sigma,matlab = self.mat)
        self.d = d
        self.ppd = ppd
        self.p = p
    
    def filterBank(self, ftype ='fDoG', **kwargs):
        '''
        Remap a bank of 2D spatial filter images to a series of one-dimensional 
        column vectors by a lexicographical index. Then combine column vectors
        into a filter bank matrix. 
        
        Parameters:
            
            filterType (type- str):
                valid key name that determines the type of filter matrix to 
                create. Valid names include: 'DoG' (difference of gaussians),
                'Gabor' (Gabor filter), 'LGauss' (Laguerre–Gauss filter).
        
        Return:
            MxN matrix where M is the number of pixels in the 2D filter image
            and N is the number of filters to be created. 
        '''
        
        if ftype not in ['fDoG','sGabor','sLGauss']: 
            er = 'This filter type, {0} has not been implemented yet here.'
            raise ValueError(er.format(ftype))
            
        elif len(kwargs.keys()) == 0:
            #default filter banks if no user defined parameters for filter are
            #specified in kwargs 
            filter_parameters = {
            'fDoG': {
                'N': 10,
                },
            'sGabor': {
                'b': [1],
                'Theta': [(i/8)*np.pi for i in range(0,8)],
                'Lambda': [self.ppd/(2**i) for i in range(5)], #convert to 
                                                           #cycles per degeree
                'Phi': [0],
                'Gamma': [1],
                },
            'sLGauss': {
                'A': [8], #[5,14,8],
                'B': [8], #[5,14,8],
                'N': 6,
                }
            }
            kwargs = filter_parameters[ftype]
            
        else:
            pass
        
        chan_matrx = []
        params = []
        chan = super().channels(ftype = ftype, **kwargs)
           
        i = 0 
        for k, v in chan.items():
            if v[0] is None:
                continue
            channel =v[0].flatten().reshape(-1,1)
            chan_matrx.append(channel)
            v[1]['column of T'] = i
            params.append(v[1])
            i += 1
            
        return(np.concatenate(chan_matrx, axis = 1), params)
    
    def analyticCov(self, channelMatrix):
        covMatrix = np.zeros(2*[channelMatrix.shape[1]])
        trial_nps = self.trial.npsv ** 2 #make sure I make a copy of this 
        for i in range(channelMatrix.shape[1]):
            ch = channelMatrix[:,i].reshape(self.d)
            #plt.imshow(ch,cmap = 'gray')
            chFiltered = np.fft.fft2(ch) * trial_nps
            #plt.imshow(np.fft.ifft2(chFiltered).real,cmap = 'gray')
            #return
            chSpatialD = np.fft.ifft2(chFiltered).real.flatten().reshape(1,-1)
            
            for j in range(channelMatrix.shape[1]):
                ch1 = channelMatrix[:,j]
                covMatrix[i,j] = chSpatialD@ch1
                
        return(covMatrix)
                
                
    def trainCov(self, Nsamples, channelMatrix):
        '''
        Train the template for the channelized hotelling observer with a 
        specific filter bank type 

        Parameters
        ----------
        Nsamples : TYPE, int.
            DESCRIPTION. Number of samples to train the CHO template.
        ftype : TYPE, optional
            DESCRIPTION. The default is 'fDoG'. But can also be 'sGab' or
            'sLGauss'

        Returns
        -------
        None. a template of CHO model. 

        '''
        TU = []
        Tt = channelMatrix.T #transpose of filter bank matrix
        trials = self.trial.sim_trials(0, Nsamples, LKE = False, flatten = True, 
                   fourier = False)

        samples = trials['images']
        
        for u in samples:
            TU.append(Tt@u) #take the matrix product of Tt and random vector u
            
        TU = np.concatenate(TU, axis = 1)
        res = TU - TU.mean(axis = 1, keepdims=True)
        cov = (res@res.T)/(Nsamples - 1)
        
        return(cov)

    def template(self, Nsamples = False, ftype = 'sGabor', **kwargs):
        
        T, params = self.filterBank(ftype = ftype, **kwargs) #channel matrix
        s = self.trial.signal.flatten().reshape(-1,1)  
        
        new_p = {
            "{0} parameters".format(ftype): params,
                  }
        
        if Nsamples != False:
            cov = self.trainCov(Nsamples, T)
        else:
            cov = self.analyticCov(T)
            
        inv = sci.inv(cov)
        v = inv@T.T@s #channel template weights
        w = (T@v).reshape(self.d) #template spatial domain
        
        new_p['K^-1'] = inv #inverse of the covariance matrix for channels
        new_p['Vhot'] = v #weight vector for filter channels 
        new_p['K'] = cov #covariance matrix of template
        new_p['Ws'] = w #template in the spatial domain 
        new_p['Wf'] = np.fft.fft2(np.fft.fftshift(w)) #template fourier domain
        return(new_p)
        
   
    
   
    
   
    
   
    
   
    def choDp(self, **kwargs):
        dp_channels = np.sqrt((kwargs['Vhot'].T@kwargs['K^-1']@kwargs['Vhot']))
        
        CW = np.abs(np.fft.ifft2((self.trial.npsv * kwargs['Wf'])).real)
        WTCW = np.sum(kwargs['Ws'] * CW) #variance of lambda 
        deltaMu_lambda = np.sum(kwargs['Ws'] * self.trial.signal) #mean of lambda
        dp = deltaMu_lambda/np.sqrt(WTCW)
        
        wf_conj = np.conj(kwargs['Wf'])
        fsig = np.fft.fft2(np.fft.ifftshift(self.trial.signal))
        numerator = np.sum((wf_conj * fsig))
        denominator = (np.sum((np.abs(kwargs['Wf'])**2 * self.trial.npsv**2)))**.5
        
        dp1 = numerator.real / denominator
        #print (numerator, denominator)
        #print(dp1)
        #print('craig help', deltaMu_lambda, WTCW)
        #print ('miguel chapter 10',numerator, denominator)
        
        res = {"dp channels":dp_channels,
               "dp": dp,
               "dpMiguelchp10": dp1,
               "var lambda": WTCW,
               "deltaMu lambda": deltaMu_lambda,
               }
        return(res)
        
    def choTemp(self, Nsamples = False, ftype = 'sGabor', **kwargs):
        template = self.template(Nsamples = Nsamples, ftype = ftype, **kwargs)
        dp = self.choDp(**template)
        return({**template,**dp})         
        
        
        
 
    
 
    
 
    
 
    
 
    
###############################################################################
#Main code below
###############################################################################       
#Example code to plot the filer in the spatial domain 
if __name__ == "__main__":
    Ns= int(1e4)
    params = [(1024,1024), #dimension of image
              2.8, #power law
              80, #signal radius
              1, #signal contrast
              45, #pixels per degree
              128, #mu,
              30, # sigma
              ]
    c = CHO(*params)

    #CHO Gabor
    a = c.choTemp()
    plt.imshow(a['Ws'], cmap = 'gray')
    #t = c.choTemp(Nsamples = Ns)
    
# =============================================================================
#     responses = []
#     for i in range(10000):
#         t = c.trial.simTrial(LKE = True, signalpresent = True, fourier = True)
#         responses.append(c.skeResponse(t, a['Wf']))
#     response = np.array(responses)
# =============================================================================
    #print('train', response.mean(), response.var(ddof = 1))
# =============================================================================
#     gabor_template = c.choTemp(Nsamples = False, ftype = 'sGabor')
#     x = np.linspace(-50,800, 10000)
#     s = norm.pdf(x, loc = gabor_template['deltaMu lambda'], scale = gabor_template['var lambda']**.5)
#     n = norm.pdf(x, loc = 0, scale = gabor_template['var lambda']**.5)
#     fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
#     ax.plot(x,s, 'r-', label = 'signal')
#     ax.plot(x,n, 'b-', label = 'noise')
# =============================================================================

    
    
    
    #im = ax.imshow(gabor_template['Ws'], cmap = 'gray')
    #ax.set_title('CHO Gabor channel Template', fontsize = 'large', fontweight = 'bold')
    #plt.colorbar(im)
    #plt.savefig('/Users/Devi/Desktop/CHO_Gabor_Template.jpg')
    #gabor_params.to_csv('/Users/Devi/Desktop/CHO Gabor parameters.csv')
    
###############################################################################
#                              D prime CHO
###############################################################################
# =============================================================================
#     #computing dprime for gabor template
#     fgab = np.fft.fft2(gabor_template['template'])
#     fgabC = np.conj(fgab)
#     fsig = np.fft.fft2(c.diskSignal(normalize = False) * params[3])
#     dp = np.sum(np.fft.ifft2(fgabC * fsig))/(np.sum(np.fft.ifft2(fgab.real**2 * c.trial.filter)))**.5
#     
#     print(dp)
# 
#     varT = np.sum(np.fft.ifft2(fgab * c.trial.filter).real)
#     
#     dp1 = np.sum(gabor_template['template']*c.diskSignal(normalize = False) * params[3])/np.sqrt(varT)
#     print(dp1)
# =============================================================================
    
    
###############################################################################
#                               Other Templates
###############################################################################
# =============================================================================
#     #CHO DoG
#     dog_template, dog_params = c.trainTemplate(Ns, ftype = 'fDoG')
#     
#     fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
#     im = ax.imshow(dog_template, cmap = 'gray')
#     ax.set_title('CHO DoG channel Template', fontsize = 'large', fontweight = 'bold')
#     plt.colorbar(im)
#     plt.savefig('/Users/Devi/Desktop/CHO_DoG_Template.jpg')
#     dog_params.to_csv('/Users/Devi/Desktop/CHO DoG parameters.csv')
# =============================================================================
    
# =============================================================================
#     #CHO Laguerre–Gauss
#     lg_template, lg_params = c.trainTemplate(Ns, ftype = 'sLGauss')
#     
#     fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
#     im = ax.imshow(lg_template, cmap = 'gray')
#     ax.set_title('CHO Laguerre–Gauss channel Template', fontsize = 'large', fontweight = 'bold')
#     plt.colorbar(im)
#     plt.savefig('/Users/Devi/Desktop/CHO_Laguerre–Gauss_Template.jpg')
#     lg_params.to_csv('/Users/Devi/Desktop/CHO Laguerre–Gauss parameters.csv')
# =============================================================================
    

    

        

