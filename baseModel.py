#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:49:31 2020

@author: Devi

Model Base class.
"""

import math 
import numpy as np 
import matplotlib.pyplot as plt

class baseModel:
    def __init__(self, dimensions, sigRad):
        '''
        This is a base class to be used for all other classes in the model 
        observer package. It works only for a disk type signal embedded in a
        2D image.
        
        Parameters
        ----------
        dimensions : TYPE, tuple of length 2.
            DESCRIPTION. The dimensions of the 2D image (rows,columns).
        sigRad : TYPE, int.
            DESCRIPTION. The radius of the disk, which is the signal in this
            model observer class for now.

        Returns
        -------
        None.

        '''
        self.dim = dimensions 
        self.sig_rad = sigRad
        self.rc = int(dimensions[0]/2)
        self.cc = int(dimensions[1]/2)
        
    def __repr__(self):
        '''
        This method is primaraly used for debugging and for the devloper to 
        see what parameters for the class were fed in to create an instance of 
        this class.

        Returns
        -------
        A string. that represents the syntax used to create this class object.

        '''
        return(f'{self.__class__.__name__}('
               f'{self.dim!r}, {self.sig_rad!r})')
    
    def __str__(self):
        param1 = f'A {self.dim[0]} row by {self.dim[1]} column image.'
        param2 = f'With a disk signal that has a radius of {self.sig_rad} pixels.'
        return ('{0}\n{1}'.format(param1,param2))
    
    def coordSpace(self, customDIM = ()):
        if len(customDIM) == 0:
            rows = self.dim[0]
            cols = self.dim[1]
        else:
            rows = customDIM[0]
            cols = customDIM[1]

        halfc = math.ceil((cols - 1)/2)
        halfr = math.ceil((rows - 1)/2)
        cols_ = np.linspace(-halfc, halfc - 1, cols)
        rows_ = np.linspace(halfr, -1*(halfr - 1), rows)
        x, y = np.meshgrid(cols_,rows_)
        return(x,y)
        
    def freqDom(self,):
        x,y = self.coordSpace()
        return (x/self.dim[1], y/self.dim[0])
    
    def distance(self, x,y):
        return (np.sqrt(x**2 + y**2))
        
    def NPS(self,f):
        '''
        This is a power law filter, created in the fourier domain. This will 
        have unit variance and is to be used to filter white noise and 
        it is used in the model observer classes.

        Parameters
        ----------
        f : TYPE, float
            DESCRIPTION. The how to modulate the frequencies, i.e. the power 
            law.

        Returns
        -------
        TYPE, 2D array.
            DESCRIPTION. 2D power law filter in the frequency domain.

        '''
        freqs = self.freqDom() #get spatial frequency templates u,v 
        dist = self.distance(*freqs) #compute the radial spatial frequency
        dist = np.fft.ifftshift(dist) #shift the DC component to top left corner
        with np.errstate(divide='ignore'): #prevent warning for dividing by zero
            temp = 1/(dist**f) #power law filter the spatial frequencies 
        temp[0,0] = temp[0,1] #replace the dc component (inf) with neighboring value
        unnorm_filter = np.sqrt(temp) #unnormailzed filter 
        m_unnorm = unnorm_filter.mean() #mean of template
        num_pix = (temp.shape[0] * temp.shape[1])
        var_filter= np.sum(np.abs(unnorm_filter - m_unnorm)**2) / num_pix
        
        if var_filter == 0:
            return(unnorm_filter)
        else:
            norm_filter = unnorm_filter / np.sqrt(var_filter) #normalize filter 
            return(norm_filter)
        
    
    def diskSignal(self, normalize = True, customRadius = False,
                   customDIM = None):
        '''
        Create a Disk shaped signal with a specific radius.

        Parameters
        ----------
        normalize : TYPE, optional
            DESCRIPTION. The default is True. Should the sum of all the values
            = 1 or not? 
        customRadius : TYPE, optional
            DESCRIPTION. The default is False. If not False, then this is type
            int and it represents the radius of the disk signal
        customDIM : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE, 2D Numpy Array.
            DESCRIPTION. a disk embedded in an array of zeros. 

        '''
        if customDIM is None:
            zeros = np.zeros(self.dim)
            x,y = self.coordSpace()
        else:
            zeros = np.zeros(customDIM)
            x,y = self.coordSpace(customDIM = customDIM)
        
        if customRadius:
            r = customRadius
        else:
            r = self.sig_rad 
        
        dist = self.distance(x, y)
        condition = dist <= r
        zeros[condition] += 1
        
        if normalize:
            zeros = zeros/zeros.sum()
        return(zeros)
    
    def tempConvolve(self, trial, template):
        '''
        Convolve the template of a model observer with the trial in the 
        fourier domain and revert back to spatial domian.

        Parameters
        ----------
        trial : TYPE, 2D array of type complex values.
            DESCRIPTION. The trial in the fourier domain.
        template : TYPE, 2D array of type complex values.
            DESCRIPTION. The tempalte in the fourier domain.

        Returns
        -------
        Convolved template with trial in the spatial domain.

        '''
        convolve = np.multiply(trial, template)
        return(np.fft.ifft2(convolve).real)
        
    def skeResponse(self, trial, template):
        '''
        Get the response of the model observer at the KNOWN location of the 
        signal (LKE experiment).

        Parameters
        ----------
        trial : TYPE, 2D array of type complex values.
            DESCRIPTION. The trial in the fourier domain.
        template : TYPE, 2D array of type complex values.
            DESCRIPTION. The tempalte in the fourier domain.

        Returns
        -------
        TYPE, float.
            DESCRIPTION. The response of the model observer at the signal
            location.

        '''
        return(np.sum(trial * template))
        #return(self.tempConvolve(trial,template)[self.rc,self.cc])
    
    

  

            
        
        
        
        
        
        
###############################################################################
#Main code below
############################################################################### 
if __name__ == "__main__":  
    params = [(128,128),
              3,
              ]
    
    vars_ = []
    var_z = []
    m = baseModel(params[0], params[1])
    
    c = m.NPS(2.8)
    
    
# =============================================================================
#     for i in range(2000):
#         z = np.random.normal(0, 1, (128,128))
#         zf = np.fft.fft2(z)
#         zff = zf *c
#         invzff = np.fft.ifft2(zff)
#         var_z.append(z.var(ddof = 1))
#         vars_.append(invzff.real.var(ddof = 1))
#     
#     print(np.array(vars_).mean())
#     print(np.array(var_z).mean())
# =============================================================================
    
    
    #d = m.diskSignal(normalize = False)
    
    
    
    
    
    #a = m.diskSignal(normalize = True, customRadius = 80, customDIM = (500,500))
# =============================================================================
#     #s1 = m.diskSignal()
#     f = 1*m.Filter(0)
#     A = np.random.normal(0,1,(100,100))
#     fA = np.fft.fft2(A)
#     Afiltered = np.fft.ifft2(np.multiply(f,fA)).real + 128
#     plt.imshow(Afiltered, cmap = 'gray')
# =============================================================================
# =============================================================================
#     img  = np.random.normal(size = params[0])
#     img[450:550,450:550] += 10
#     #print(repr(m))
#     
#     U, V = m.freqDom()
#     Z = m.modulusC()
#     #print("column values",U)
#     #print("row values", V)
#     h = plt.contourf(U,V,Z, cmap = 'gray')
#     plt.show()
#     
#     plt.imshow(img,cmap = 'gray')
#     
#     
#     filtered = np.fft.ifft2(np.multiply(Z,np.fft.fft2(img))).real
#     plt.imshow(filtered,cmap = 'gray')
# =============================================================================

