from scipy import io
import cv2, os
import math
import numpy as np
import transforms3d

from constants import *

def loadFile(file):
	x = io.loadmat(file)
	return x

def showImage(img, imageName='Image'):
	cv2.imshow(imageName,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()