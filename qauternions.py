from scipy import io
import cv2, os
import math
import numpy as np

def quatAdd(a, b):
	np.add(a, b)

def quatMultiply(a, b):
	afirst = a[0, 0]
	asecond = np.asarray([a[0, 1], a[0, 2], a[0, 3]])

	bfirst = b[0, 0]
	bsecond = np.asarray([b[0, 1], b[0, 2], b[0, 3]])

	resFirst = afirst * bfirst - 

def quatConjugate(a):
	pass