
import numpy
import cv2
import sys
import os



PATH=''


def normalise(data):
    data=data.astype(float)
    n=numpy.where(numpy.isnan(data)==False)
    dt = data[n[0],n[1]]
    mn = dt.min()
    mx = dt.max()
    normed = (data-mn)/(mx-mn)
    
    return numpy.where(numpy.isnan(normed)==True,0.0,normed)
            
def createVideo(images,fname,path=PATH,ext='.mov',frameRate=30, x1=0, x2=None, y1=0, y2=None, nm=0, codec='mp4v'):
   
    if images.ndim==4:
        n,h,w,l=images.shape
    
    elif images.ndim==3:
        n,h,w=images.shape
    
    if type(y1)==type(None):
        y1=0
    if type(x1)==type(None):
        x1=0
    if type(y2)==type(None):
        y2=h
    if type(x2)==type(None):
        x2=w
        
    w=x2-x1
    h=y2-y1
    
    video=cv2.VideoWriter(os.path.normpath(os.path.join(path, fname+ext)),cv2.VideoWriter_fourcc(*'%s'%codec),frameRate,(w,h))

    for i in range(images.shape[0]):
#        sys.stdout.write('\rProcessing frame %i of %i'%(i+1,images.shape[0]))
#        sys.stdout.flush()
        
        if images.ndim == 4:
            img = numpy.zeros(images[i].shape)
            img[:,:,0] = images[i][:,:,2]
            img[:,:,2] = images[i][:,:,0]
            img[:,:,1] = images[i][:,:,1]
        else:
            img = numpy.zeros((images[i].shape[0], images[i].shape[1], 3))
            if nm == 1:
                img[:,:,0] = normalise(images[i])*255
                img[:,:,2] = normalise(images[i])*255
                img[:,:,1] = normalise(images[i])*255
            else:
                img[:,:,0] = images[i]
                img[:,:,2] = images[i]
                img[:,:,1] = images[i]
        video.write(img[y1:y2, x1:x2].astype(numpy.uint8))
    
#    sys.stdout.write('\n')
#    sys.stdout.flush()
    video.release()
    return