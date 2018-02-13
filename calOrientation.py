from scipy import io
import cv2, os
import math
import numpy as np

from constants import *
from plots import *
from utils import *
from kalmanFilter import *
import kalmanFilter2 as kf2
import pdb

def seeCameraSample():
	data = loadFile(os.path.join(CAM_FOLDER, 'cam1.mat'))
	data = data['cam']
	shapeOfData = data.shape
	for i in range(shapeOfData[3]):
		cv2.imshow('Image',data[:, :, :, i])
		cv2.waitKey(10)
	cv2.destroyAllWindows()

def rawAccToPhysical(ax, ay, az):
	biasX = 510 #1510 #mV
	biasY = 501 #1490
	biasZ = 504 #606 #1510
	sensitivityX = 330 #mV/g
	sensitivityY = 330
	sensitivityZ = 330
	vref = 3300 #mV
	rx = ((((ax - biasX) * vref) / 1023.0)) / sensitivityX
	ry = ((((ay - biasY) * vref) / 1023.0)) / sensitivityY
	rz = ((((az - biasZ) * vref) / 1023.0)) / sensitivityZ
	return -1 * rx, -1 * ry, rz  	# negates the rx, ry because accelerometer reading is opposite

def rawAngularVelToPhysical(wx, wy, wz):
	biasX = 373 #1230 #mV
	biasY = 376 #1230
	biasZ = 370 #1230
	sensitivityX = 3.33 #mV/g
	sensitivityY = 3.33			# confirm!
	sensitivityZ = 3.33
	vref = 3300 #mV
	rx = ((((wx - biasX) * vref) / 1023.0)) / sensitivityX
	ry = ((((wy - biasY) * vref) / 1023.0)) / sensitivityY
	rz = ((((wz - biasZ) * vref) / 1023.0)) / sensitivityZ
	toRadian = math.pi / 180
	rx, ry, rz = rx * toRadian, ry * toRadian, rz * toRadian
	return rx, ry, rz

def checkAccSensorData():
	imuFileName = 'imuRaw7.mat'
	viconFileName = 'viconRot7.mat'

	data = loadFile(os.path.join(IMU_FOLDER, imuFileName))
	sensorData = data['vals']
	timestamps = data['ts']
	numInstances = timestamps.shape[1]
	numInstances = 200
	# pdb.set_trace()

	accelData = np.zeros((numInstances, 3))

	for i in range(numInstances):
		ax, ay, az = rawAccToPhysical(sensorData[0, i], sensorData[1, i], sensorData[2, i])
		accelData[i, 0] = ax
		accelData[i, 1] = ay
		accelData[i, 2] = az
		print accelData[i, :]

def applyFilterAndCompare():
	imuFileName = 'imuRaw9.mat'
	viconFileName = 'viconRot9.mat'
	camFileName = 'cam9.mat'

	data = loadFile(os.path.join(IMU_FOLDER, imuFileName))
	sensorData = data['vals']
	timestamps = data['ts']
	numInstances = timestamps.shape[1]

	gyroData = np.zeros((numInstances, 3))
	accelData = np.zeros((numInstances, 3))

	# print np.sum(sensorData[:, 0:200], 1)/200

	for i in range(numInstances):
		wx, wy, wz = rawAngularVelToPhysical(sensorData[4, i], sensorData[5, i], sensorData[3, i])
		gyroData[i, 0] = wx
		gyroData[i, 1] = wy
		gyroData[i, 2] = wz

		ax, ay, az = rawAccToPhysical(sensorData[0, i], sensorData[1, i], sensorData[2, i])
		accelData[i, 0] = ax
		accelData[i, 1] = ay
		accelData[i, 2] = az

	# filterResult = UKF(gyroData, accelData, timestamps)
	filterResult = kf2.UKF(gyroData, accelData, timestamps)
	# filterResult = checkGyroIntegration(gyroData, timestamps)
	plotGTruthAndPredictions(viconFileName, filterResult, timestamps)
	# plotPredictions(filterResult, timestamps)
	# createPanoramaFromPredictions(timestamps, filterResult, camFileName)


if __name__ == "__main__":
	# data = loadFile(os.path.join(CAM_FOLDER, 'cam1.mat'))
	# print data
	# seeCameraSample()
	# viewVicon()
	# plotEulerAnglesVicon()
	# checkAccSensorData()
	# createPanoramaFromViconData()
	applyFilterAndCompare()