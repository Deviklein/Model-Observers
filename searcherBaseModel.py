#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:06:59 2020

@author: Devi
"""
from baseModel import baseModel
from scipy.stats import norm
import numpy as np

class BaseSearcher(baseModel):
    def __init__(self, dimensions, sigRad, ppd):
        super().__init__(dimensions, sigRad)
        self.ppd = ppd
        self.fovea = super().diskSignal(normalize = False, customRadius = ppd)
    
    def logLHRMap(self, trial, df, modelTemplate, ecc):
        assert (trial.dtype == 'complex128')
        assert (modelTemplate.dtype == 'complex128')
        search = self.tempConvolve(trial, modelTemplate)
        muP = df[(df['Eccentricity'] == ecc) & (df['ground truth'] == 1)]['mean']
        muA = df[(df['Eccentricity'] == ecc) & (df['ground truth'] == 0)]['mean']
        stdP = df[(df['Eccentricity'] == ecc) & (df['ground truth'] == 1)]['std']
        stdA = df[(df['Eccentricity'] == ecc) & (df['ground truth'] == 0)]['std']
        logP = norm.logpdf(search, loc=muP, scale=stdP) 
        logA = norm.logpdf(search, loc=muA, scale=stdA)
        return ({ecc :  logP - logA})
    
    def eccMap(self, numecci, dim = (512,512)):
        emap = np.zeros(dim) + numecci
        for i in range(numecci, 0, -1):
            temp = super().diskSignal(normalize = False,
                                      customRadius = i * self.ppd, 
                                      customDIM = dim,
                                      )
            emap -= temp
        return(emap)
    
    def FillEyeField(self, arr1, arr2, ecc):
        inds = arr1 == ecc
        arr1[inds] = arr2[inds]
    
        
    def moveEye(self, row, col, arr, filler):   
        rc, cc = int(arr.shape[0]/2), int(arr.shape[1]/2)
        distr, distc = row - rc, col - cc
                
        newfix = np.roll(arr,(distr, distc), axis = (0,1))
        
        if (distr < 0) and (distc < 0):
            newfix[distr:] = filler
            newfix[:,distc:] = filler
            return(newfix)
        elif (distr < 0) and (distc > 0):
            newfix[distr:] = filler
            newfix[:,:distc] = filler
            return(newfix)
        elif (distr > 0) and (distc < 0):
            newfix[:distr] = filler
            newfix[:,distc:] = filler
            return(newfix)
        elif (distr > 0) and (distc > 0):
            newfix[:distr] = filler
            newfix[:,:distc] = filler
            return(newfix)
        elif (distr == 0) and (distc < 0):
            newfix[:,distc:] = filler
            return(newfix)
        elif (distr == 0) and (distc > 0):
            newfix[:,:distc] = filler
            return(newfix)
        elif (distr < 0) and (distc == 0):
            newfix[distr:] = filler
            return(newfix)
        elif (distr > 0) and (distc == 0):
            newfix[:distr] = filler
            return(newfix)
        else:
            return(newfix)
    
    def fixHist(self, array, locRow, locCol,):
        newHist = array + self.moveEye(locRow,locCol,self.fovea, 0)
        return(newHist)
    
        
        