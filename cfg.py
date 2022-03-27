# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:38:54 2019
simpan model dan traning dalam file

@author: user
"""
import os

class Config:
    def __init__ (self, mode='time', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')