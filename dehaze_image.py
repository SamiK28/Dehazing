import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Enter path to the image")
args = vars(ap.parse_args())

src=cv2.imread(args["image"])

processed=dehaze(src)

cv2.imshow("Histogram Equalized",processed)
cv2.imshow("Original",src)
cv2.waitKey(0)
cv2.destroyAllWindows()