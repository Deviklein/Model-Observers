#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:11:39 2020

@author: Devi
"""

from modelObserverSpace import space, nps, signal
#from modelObserverSpace import background, trial
from scipy.special import comb, factorial
#from scipy.stats import norm
import numpy as np 
import matplotlib.pyplot as plt
import scipy.linalg as sci
#import pandas as pd
import os

class Channels(space):
    def __init__(self, dimensions):
        '''
        Initialize the 2D channels for the CHO model.

        Parameters
        ----------
        dimensions : TYPE, tuple of length 2.
            DESCRIPTION. Dimensions of the image (rows,columns)

        Returns
        -------
        None.

        '''
        super().__init__(dimensions)
    
    def normChannels(self, channel):
        '''
        Normalize the values in the 2D channel template. 

        Parameters
        ----------
        channel : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
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
        freqs = super().twoD_radialFreq() #2D spatial frequency domain 
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
        X, Y = super.twoD_Euclid()
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
        x, y = super().twoD_Euclid()

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

class CHO(nps):
    def __init__(self, d, p, sr, c, ppd, sigma, fp_nps = None):
        super().__init__(d, p,)
        if fp_nps == None:
            self.nps = (sigma * super().nps2D())**2
        else:
            self.nps = super().loadnps(fp_nps, twoD = True)
        self.d = d
        self.ppd = ppd
        self.sig = signal(d, sr, c,)
        self.chnnls = Channels(d)

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
            
        if len(kwargs.keys()) == 0:
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
        chan = self.chnnls.channels(ftype = ftype, **kwargs)
           
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
    
    def analyticCov(self, channelMatrix,):
        covMatrix = np.zeros(2*[channelMatrix.shape[1]])
        for i in range(channelMatrix.shape[1]):
            ch = channelMatrix[:,i].reshape(self.d[:2])
            chFiltered = np.fft.fft2(ch) * self.nps
            chSpatialD = np.fft.ifft2(chFiltered).real.flatten().reshape(1,-1)
            
            for j in range(channelMatrix.shape[1]):
                ch1 = channelMatrix[:,j]
                covMatrix[i,j] = chSpatialD@ch1
                
        return(covMatrix)

    def twoDTemplate(self, ftype = 'sGabor', sigtype = 'mass', **kwargs): 
        assert len(self.d) == 2
        
        if sigtype == 'mass':
            sig = self.sig.mass()
        elif sigtype == 'microcalc':
            sig = self.sig.calc()
        else:
            er = 'This signal type, {0} has not been implemented yet here.'
            raise ValueError(er.format(sigtype))
        T, params = self.filterBank(ftype = ftype, **kwargs) #channel matrix
        cov = self.analyticCov(T)
        s = sig.flatten().reshape(-1,1)  
        
        new_p = {
            "{0} parameters".format(ftype): params,
                  }
            
        inv = sci.inv(cov)
        v = inv@T.T@s #channel template weights
        w = (T@v).reshape(self.d[:2]) #template spatial domain
        
        new_p['K^-1'] = inv #inverse of the covariance matrix for channels
        new_p['Vhot'] = v #weight vector for filter channels 
        new_p['K'] = cov #covariance matrix of template
        new_p['Ws'] = w #template in the spatial domain 
        new_p['Wf'] = np.fft.fft2(np.fft.fftshift(w)) #template fourier domain
        return(new_p)
    
    def threeDTemplate(self, ftype = 'sGabor', sigtype = 'mass', **kwargs):
        assert len(self.d) == 3
        
        T, params = self.filterBank(ftype = ftype, **kwargs) #channel matrix
        cov = self.analyticCov(T)
        inv = sci.inv(cov)
        
        if sigtype == 'mass':
            sig = self.sig.mass()
        elif sigtype == 'microcalc':
            sig = self.sig.calc()
        else:
            er = 'This signal type, {0} has not been implemented yet here.'
            raise ValueError(er.format(sigtype))
        
        temp_arr = np.zeros(self.d)
        for i in range(self.d[-1]):
            s = sig[:,:,i].flatten().reshape(-1,1) 
            v = inv@T.T@s #channel template weights
            w = (T@v).reshape(self.d[:2]) #template spatial domain
            temp_arr[:,:,i] = w
        
        return(temp_arr)

class fCHO(CHO):
    def __init__(self, d, p, sr, c, ppd, sigma, signaltype, fp_nps = None):
        super().__init__(d, p, sr, c, ppd, sigma, fp_nps = fp_nps)
        self.ppd = ppd
        self.d = d
        self.sigtype = signaltype
    
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
        if len(self.d) == 3:
            return(super().threeDTemplate(ftype = 'sGabor',
                                          sigtype = self.sigtype, **kwargs))
        elif len(self.d) == 2:
            temp = super().twoDTemplate(ftype = 'sGabor', 
                                        sigtype = self.sigtype, **kwargs)
            return(temp['Ws'])
    
    def tempEccBank(self,numecci):
        temps = {}

        for ecc in range(0,numecci+1):
            t = self.tempEcc(ecc)
            temps["eccentricity {0}".format(ecc)] = t
            
        return (temps)

class Eye(signal):
    def __init__(self, d, ppd,):
        self.d = d
        self.ppd = ppd
        
    def eccMap(self, numecci,):
        emap = np.zeros(self.d) + numecci
        for i in range(numecci, 0, -1):
            temp = super().__init__(dimensions = self.d, 
                                    signalradius = i * self.ppd,)
            
            emap -= temp.calc()
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




if __name__ == "__main__":  

    ##############################################################################
    #Testing the code below
    ##############################################################################        
# =============================================================================
#     #CHO Test
#     params = [
#         (128,128,10),
#         2.8,
#         3,
#         50,
#         45,
#         30,  
#         ]
#     c = CHO(*params)
#     template = c.threeDTemplate(sigtype = 'microcalc')  
#     
#     for i in range(params[0][-1]):
#         plt.figure()
#         plt.title("slice # {0}".format(i))
#         plt.imshow(template[:,:,i], cmap = 'gray')
#     #plt.close('all')
# =============================================================================
    
    #fCHO test
    
    params_mass = [
        (820,1024,100),
        2.8,
        3,
        83,
        45,
        30,  
        'mass',
        os.path.join(os.getcwd(),"powerSpectrum-820-1024.mat"),
        #"/Users/Devi/Desktop/fCHO_3D/powerSpectrum-820-1024.mat",
        ]
    
    params_micro = [
        (820,1024,100),
        2.8,
        3,
        83,
        45,
        30, 
        'microcalc',
        os.path.join(os.getcwd(),"powerSpectrum-820-1024.mat"),
        #"/Users/Devi/Desktop/fCHO_3D/powerSpectrum-820-1024.mat",
        ]
    
    
    fps = [
        os.path.join(os.getcwd(),'templates/mass'),
        os.path.join(os.getcwd(),'templates/micro')
        #'/Users/Devi/Desktop/fCHO_3D/templates/mass',
        #'/Users/Devi/Desktop/fCHO_3D/templates/micro',
           ]
    sigs = [
        fCHO(*params_mass),
        fCHO(*params_micro),
            ]
    sig_names = [
        'mass',
        'microcalc'
        ]
    
    for ind, fp in enumerate(fps):
        if not os.path.exists(fp):
            os.makedirs(fp)

        fp_out = os.path.join(fp,"alleccentricity_{0}".format(sig_names[ind]))
        templates = sigs[ind].tempEccBank(9)
        np.savez(fp_out, **templates)
            
    
    #plotting the templates at each eccentricity
# =============================================================================
#     npzfile = np.load('/Users/Devi/Desktop/fCHO_3D/templates/mass/alleccentricity_mass.npz')
#     npzfile1 = np.load('/Users/Devi/Desktop/fCHO_3D/templates/microcalc/alleccentricity_microcalc.npz')
#     for i in range(28,29):#73):
#         fig, axes = plt.subplots(ncols = 4, nrows = 3)
#         fig.suptitle('Slice: {0}'.format(i))
#         for ind, ax in enumerate(axes.flat):
#             if ind > 10:
#                 continue
#             ax.set_title("Eccentricity {0}".format(ind))
#             ax.imshow(npzfile['eccentricity {0}'.format(ind)][:,:,i], cmap = 'gray')
#         plt.tight_layout()
# =============================================================================
        