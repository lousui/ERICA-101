#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:29:27 2023

@author: laurayoung-adapted from Penny Lawton
"""

import numpy


def riemann(imagedata):
    arr=1+(numpy.gradient(imagedata)[0])**2+((numpy.gradient(imagedata)[1])**2-(numpy.gradient(imagedata)[0])**2)*(numpy.gradient(imagedata)[1])**2
    S=(1/(imagedata.shape[0]*imagedata.shape[1]))*numpy.nansum(abs(arr))
    return S,arr

def sqlaplace(imagedata):
    
    arrl=numpy.gradient(imagedata)
    laplace = ((numpy.gradient(arrl[0])[0])+(numpy.gradient(arrl[1])[1]))**2
    S=numpy.nansum(laplace)
    
    return S,laplace

def mlv(data,x,y):
    a=numpy.array((data[x+1,y],data[x-1,y],data[x,y+1],data[x,y-1],data[x+1,y+1], data[x-1,y-1],data[x+1,y-1],data[x-1,y+1]))
    b=data[x,y]*numpy.ones((8))
    c=abs(b-a)
    d=max(c)
    return d

def mlv_image(imagedata):
    length = imagedata.shape[0]
    width = imagedata.shape[1]
    
    mlvmap = numpy.zeros((length,width))
    for idx,x in numpy.ndenumerate(imagedata):
            mlvmap[idx]=mlv(imagedata,idx[0]-1,idx[1]-1)
            
    mlvmap=mlvmap[1:length-1,1:width-1]
    
            
    return mlvmap

