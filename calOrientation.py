from scipy import io
import cv2, os
import math
import numpy as np

from constants import *
from plots import *
from utils import *

def seeCameraSample():
	data = loadFile(os.path.join(CAM_FOLDER, 'cam1.mat'))
	data = data['cam']
	shapeOfData = data.shape
	for i in range(shapeOfData[3]):
		cv2.imshow('Image',data[:, :, :, i])
		cv2.waitKey(10)
	cv2.destroyAllWindows()

def rawAccToPhysical(ax, ay, az):
	biasX = 1510 #mV
	biasY = 1490
	biasZ = 1510
	sensitivityX = 304 #mV/g
	sensitivityY = 306
	sensitivityZ = 300
	vref = 3300 #mV
	rx = (((ax * vref) / 1023.0) - biasX) / sensitivityX
	ry = (((ay * vref) / 1023.0) - biasY) / sensitivityY
	rz = (((az * vref) / 1023.0) - biasZ) / sensitivityZ
	return -1 * rx, -1 * ry, rz  	# negates the rx, ry because accelerometer reading is opposite

def rawAngularVelToPhysical(wx, wy, wz):
	biasX = 1230 #mV
	biasY = 1230
	biasZ = 1230
	sensitivityX = 333 #mV/g
	sensitivityY = 333			# confirm!
	sensitivityZ = 333
	vref = 3300 #mV
	rx = (((wx * vref) / 1023.0) - biasX) / sensitivityX
	ry = (((wy * vref) / 1023.0) - biasY) / sensitivityY
	rz = (((wz * vref) / 1023.0) - biasZ) / sensitivityZ
	toRadian = math.pi / 180
	rx, ry, rz = rx * toRadian, ry * toRadian, rz * toRadian
	return rx, ry, rz

if __name__ == "__main__":
	# data = loadFile(os.path.join(CAM_FOLDER, 'cam1.mat'))
	# print data
	# seeCameraSample()
	# viewVicon()
	plotEulerAnglesVicon()