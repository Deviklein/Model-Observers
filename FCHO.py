#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:11:45 2020

@author: Devi

This class is the Foveated Chanellized Hotelling Observer. It will use a gabor
filter bank for the channels to mimic the primate visual cortex. 
"""

from CHO import CHO
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

class FCHO(CHO):
    def __init__(self, d, p, sr, c, ppd, mu, sigma):
        super().__init__(d, p, sr, c, ppd, mu, sigma)
        self.ppd = ppd
        self.d = d
        
    def tempEcc(self, ecci, a = 0.7063, b =  1.6953, **kwargs):
        c = (1 + a * (ecci)**b)
        #convert wavelength of sinusoid factor to cycles per degree and scale by c
        lam = [c * self.ppd / (2**i) for i in range(5)]

        if len(kwargs) == 0:
            kwargs = {
                 'b': [1],
                 'Theta': [(i/8)*np.pi for i in range(0,8)],
                 'Lambda': lam,
                 'Phi': [0],
                 'Gamma': [1],
                }
        return(super().template(Nsamples = False, ftype = 'sGabor', **kwargs))
        #return(super().choTemp(Nsamples = False, ftype = 'sGabor', **kwargs))
    
    def tempEccBank(self,numecci):
        temps = []

        for ecc in range(0,numecci+1):
            t = self.tempEcc(ecc)
            t["eccentricity"] = ecc
            temps.append(t)
            
        return (temps)
    
    def eccMap(self, numecci, dim = (1024,1024)):
        emap = np.zeros(dim) + numecci
        for i in range(numecci, 0, -1):
            temp = super().diskSignal(normalize = False,
                                      customRadius = i * self.ppd,#/2, 
                                      customDIM = dim,
                                      )
            emap -= temp
        return(emap)
    
    def moveEye(self, row, col, **kwargs):       
        fixcenter = self.eccMap(**kwargs)
        rc, cc = int(fixcenter.shape[0]/2), int(fixcenter.shape[1]/2)
        distr, distc = row - rc, col - cc
        
        fixmax = fixcenter.max()
        newfix = np.roll(fixcenter,(distr, distc), axis = (0,1))
        
        if (distr < 0) and (distc < 0):
            newfix[distr:] = fixmax
            newfix[:,distc:] = fixmax
            return(newfix)
        elif (distr < 0) and (distc > 0):
            newfix[distr:] = fixmax
            newfix[:,:distc] = fixmax
            return(newfix)
        elif (distr > 0) and (distc < 0):
            newfix[:distr] = fixmax
            newfix[:,distc:] = fixmax
            return(newfix)
        elif (distr > 0) and (distc > 0):
            newfix[:distr] = fixmax
            newfix[:,:distc] = fixmax
            return(newfix)
        elif (distr == 0) and (distc < 0):
            newfix[:,distc:] = fixmax
            return(newfix)
        elif (distr == 0) and (distc > 0):
            newfix[:,:distc] = fixmax
            return(newfix)
        elif (distr < 0) and (distc == 0):
            newfix[distr:] = fixmax
            return(newfix)
        elif (distr > 0) and (distc == 0):
            newfix[:distr] = fixmax
            return(newfix)
        else:
            return(newfix)
        
    def respMap(self, arr1, arr2, ecc):
        inds = arr1 == ecc
        arr1[inds] = arr2[inds]
    
    def logLH(self, arr, mu, sigma):
        return(norm.logpdf(arr, loc = mu, scale = sigma))
    
    
    def foveatedResp(self, tempBank, trial, eyePos,dfMuSig):
        eyeField = self.moveEye(row = eyePos[0], col = eyePos[1])
        for i in range(len(tempBank)):
            resp = self.tempConvolve(trial, tempBank[i]['Wf'])
            mu1 = tempBank[i]["deltaMu lambda"]
            var = tempBank[i]["var lambda"]
            lr = self.logLH(resp, mu1, var) - self.logLH(resp, 0, var)
            self.respMap(eyeField, lr, tempBank[i]['eccentricity'])
        return(eyeField)
    

###############################################################################
#Main code below
###############################################################################       
#Example code to test eccentrcities of CHO model
if __name__ == "__main__":
    #for j in [20,50]:
    params = [#trial and model parameters for simulation 
              (1024,1024), #image dimensions (number rows, number columnsa)
              0, #filter noise
              3, #Disk signal profile radius 
              15, #Signal contrast
              45, #Pixels per degree
              128, #background level
              30, #standard deviation of noise, sigma 
             ]
    
    fcho = FCHO(*params)
    b = fcho.eccMap(9, dim = (1024,1024))
    
# =============================================================================
#     kwargs = {
#         "numecci": 10,
#         "dim": (512,512)
#         }
#     a = fcho.moveEye(row = 300, col = 400, **kwargs)
#     
#     b = fcho.tempEcc(5, a = 0.7063, b =  1.6953,**kwargs)
# =============================================================================
    #temp1 = fcho.tempEcc(8)
    #temp1["eccentricity"] = 8 - 1
    
# =============================================================================
#     results = []
#     for i in range(1000):
#         t = fcho.trial.simTrial(LKE = True, signalpresent = True, fourier = False)
#         results.append((temp1['Ws'] * t).sum())
# =============================================================================
    
    #plt.imshow(temp['Ws'],cmap = 'gray')
    
    #sio.savemat("/Users/Devi/Desktop/Eccentricity 8.mat",temp)
    #temps = fcho.tempEccBank(1)
    
# =============================================================================
#     trial = fcho.trial.simTrial(LKE = False, signalpresent = True, fourier = True)
#     foveated_response = fcho.foveatedResp(temps, trial, (512,512))
#     
#     fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,15))
#     axes[0].imshow(foveated_response, cmap = 'gray')
#     axes[1].imshow(np.fft.ifft2(trial).real, cmap = 'gray')
# =============================================================================
    
    
    
    
    
    #a = fcho.eccMap(10)
    #plt.imshow(a, cmap = 'gray')
    
    #b = fcho.moveEye(500, 500)
    #plt.imshow(b, cmap = 'gray')
# =============================================================================
#         for i in range(int(1e4)):
#             t = {'trial': fcho.trial.simTrial(LKE = False, signalpresent = False, fourier = True)}
#             fp = "/Users/Devi/Desktop/FCHO/SR {0}/train images/{1}".format(params[2],i)
#             sio.savemat(fp, t)
# =============================================================================
# =============================================================================
#     for t in temps:
#         t['trial parameters'] = params
#         fn = "Eccentricity {0}.mat".format(t["eccentricity"])
#         fp = "/Users/Devi/Desktop/FCHO/SR {0}/{1}".format(params[2],fn)
#         sio.savemat(fp, t)
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #plt.imshow(temps[0][7][0])
# =============================================================================
#     fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (30,30))
#     for i, ax in enumerate(axes.flat):
#         if i > 3:
#             continue 
#         temp = temps[0][i][0]
#         
#         im = ax.imshow(temp, cmap = 'gray')
#         ax.set_title('FCHO ecc {0}'.format(temps[1][i]), fontsize = 'large', fontweight = 'bold')
#         plt.colorbar(im, ax = ax)
#         plt.tight_layout()
#         plt.savefig('/Users/Devi/Desktop/FCHO first 4 tmeplates radius 20.jpg')
# =============================================================================
    