import cv2
import argparse
import math
import numpy as np

from skimage import exposure
from skimage.filters import rank


def dehaze(frame):
    b,g,r=cv2.split(frame)
    adapt = exposure.equalize_adapthist(frame, clip_limit=0.5)
    eq1=cv2.equalizeHist(b)
    eq2=cv2.equalizeHist(g)
    eq3=cv2.equalizeHist(r)
    clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(4,4))
    c1 = clahe.apply(b)
    c2 = clahe.apply(g)
    c3 = clahe.apply(r)
    eq=cv2.merge([eq1,eq2,eq3])
    cl=cv2.merge([c1,c2,c3])
    final_image=np.average([np.array(eq),np.array(cl),np.array(adapt)],axis=0,weights=[6,1,3])
    final_image=final_image.astype(np.uint8)
    return final_image


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def guided(frame):
    I = frame.astype('float64')/255;
 
    dark = DarkChannel(I,39);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,39);
    t = TransmissionRefine(frame,te);
    J = Recover(I,t,A,0.1)
    return J


cap = cv2.VideoCapture('./videos/sample video31.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==True:
        frame = cv2.resize(frame, (frame.shape[1]//1, frame.shape[0]//1), fx = 0, fy = 0, 
                         interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed=dehaze(frame)
        guided_image=guided(frame)
        cv2.imshow('histogram equalized',processed)
        cv2.imshow('dark channel prior',guided_image)
        cv2.imshow('original',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("ok")
        break

cap.release()
cv2.destroyAllWindows()
