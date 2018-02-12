from scipy import io
import cv2, os
import math
import numpy as np
import transforms3d
import pdb
from time import sleep
from skimage import img_as_ubyte

from constants import *

def loadFile(file):
	x = io.loadmat(file)
	return x

def showImage(img, imageName='Image'):
	cv2.imshow(imageName,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def findClosestIndexAfterValue(arr, val):
	pass

def getSynchArrays(gt, pred):
	gtSynch = []
	predSync = []

	gtIndex = 0
	predIndex = 0

	while True:
		if gtIndex + 1 >= gt.size or predIndex + 1 >= pred.size:
			break

		if gt[0, gtIndex] < pred[0, predIndex]:
			if gt[0, gtIndex + 1] < pred[0, predIndex]:
				gtIndex = gtIndex + 1
			else:
				if abs(gt[0, gtIndex] - pred[0, predIndex]) < abs(gt[0, gtIndex + 1] - pred[0, predIndex]):
					gtSynch.append(gtIndex)
					predSync.append(predIndex)
					gtIndex = gtIndex + 1
					predIndex = predIndex + 1
				else:
					gtSynch.append(gtIndex + 1)
					predSync.append(predIndex)
					gtIndex = gtIndex + 2
					predIndex = predIndex + 1
		else:
			if pred[0, predIndex + 1] < gt[0, gtIndex]:
				predIndex = predIndex + 1
			else:
				if abs(gt[0, gtIndex] - pred[0, predIndex]) < abs(gt[0, gtIndex] - pred[0, predIndex+1]):
					gtSynch.append(gtIndex)
					predSync.append(predIndex)
					gtIndex = gtIndex + 1
					predIndex = predIndex + 1
				else:
					gtSynch.append(gtIndex + 1)
					predSync.append(predIndex)
					gtIndex = gtIndex + 1
					predIndex = predIndex + 2

	return gtSynch, predSync

def spher2cart(r, alt, azim):
	x = r * math.cos(alt) * math.cos(azim)
	y = r * math.cos(alt) * math.sin(azim)
	z = r * math.sin(alt)
	return x, y, z

def cart2spherWithR1(x, y, z):
	norm = math.sqrt(x**2 + y**2 + z**2)
	r = 1
	alt = math.asin(z/norm)
	azim = math.atan(y/x)
	return r, alt, azim

def createPanoramaFromViconData():
	viconFileName = 'viconRot9.mat'
	viconData = loadFile(os.path.join(VICON_FOLDER, viconFileName))
	viconTs = viconData['ts']
	viconMatrices = viconData['rots']

	camData = loadFile(os.path.join(CAM_FOLDER, 'cam9.mat'))
	camDataImg = camData['cam']
	camDataTs = camData['ts']

	vt, ct = getSynchArrays(viconTs, camDataTs)
	numPoints = len(ct)

	newTs = camDataTs[0, :][ct].reshape(numPoints, 1)
	newCamDataImg = camDataImg[:,:,:,ct]
	newViconMat = viconMatrices[:,:,vt]

	imgShape = camDataImg[:,:,:,0].shape
	nrows = imgShape[0]
	ncols = imgShape[1]

	finalImage = np.zeros((1000, 1600, 3), dtype=np.uint8)

	for t in range(numPoints):
		img = newCamDataImg[:,:,:,t]
		rmat = newViconMat[:,:,t]

		nic1 = np.transpose(np.indices((nrows,ncols)), (1, 2, 0))

		nic2 = np.zeros((nic1.shape[0], nic1.shape[1], 3))
		nic2[:,:,0] = 1
		nic2[:,:,1] = ((nrows/2.0) - nic1[:,:,0]) * ((math.pi/4)/nrows)
		nic2[:,:,2] = (nic1[:,:,1] - (ncols/2.0)) * ((math.pi/3)/ncols)

		nic3 = np.zeros((nic1.shape[0], nic1.shape[1], 3))
		nic3[:,:,0] = np.multiply(np.cos(nic2[:,:,1]), np.cos(nic2[:,:,2]))
		nic3[:,:,1] = np.multiply(np.cos(nic2[:,:,1]), np.sin(nic2[:,:,2]))
		nic3[:,:,2] = np.sin(nic2[:,:,1])

		nic3 = nic3.reshape(nic1.shape[0], nic1.shape[1], 3, 1)
		nic4 = np.matmul(rmat, nic3)
		nic4 = nic4.reshape(nic1.shape[0], nic1.shape[1], 3)

		nic5 = np.zeros((nic1.shape[0], nic1.shape[1], 3))
		nic5[:,:,0] = 1
		nic5[:,:,1] = np.arcsin(nic4[:,:,2])
		nic5[:,:,2] = np.arctan(np.divide(nic4[:,:,1], nic4[:,:,0]))
		# Global spherical coordinates calculated

		nic6 = np.zeros((nic1.shape[0], nic1.shape[1], 2))
		nic6[:,:,0] = (999/2.0) - (nic5[:,:,1] * 300)
		nic6[:,:,1] = (1599/2.0) + (nic5[:,:,2] * 450)
		nic6 = np.rint(nic6).astype(int)

		img = np.fliplr(img)
		finalImage[nic6[:,:,0], nic6[:,:,1], :] = img

		cv2.imshow('Image',finalImage)
		cv2.waitKey(10)
	cv2.destroyAllWindows()

def quat2rot(q):
	q = q.reshape(4, 1)
	q = q/(np.linalg.norm(q))

	r = q[0, 0]
	i = q[1, 0]
	j = q[2, 0]
	k = q[3, 0]
	s = 1

	R = np.zeros((3, 3))
	R[0, 0] = 1 - 2*s*(j*j + k*k)
	R[0, 1] = 2*s*(i*j-k*r)
	R[0, 2] = 2*s*(i*k+j*r)

	R[1, 0] = 2*s*(i*j+k*r)
	R[1, 1] = 1 - 2*s*(i*i+k*k)
	R[1, 2] = 2*s*(j*k-i*r)

	R[2, 0] = 2*s*(i*k-j*r)
	R[2, 1] = 2*s*(j*k+i*r)
	R[2, 2] = 1 - 2*s*(i*i+j*j)

	return R

def createPanoramaFromPredictions(inputTimestamps, predcitedQuaternions, cameraFilename):
	# convert quaternions to matrices (predOrientationMatrices) indexed by time in the last component
	predOrientationMatrices = np.zeros((3, 3, inputTimestamps.size))
	for i, q_orientation in enumerate(predcitedQuaternions):
		predOrientationMatrices[:, :, i] = quat2rot(q_orientation[0:4, 0])

	camData = loadFile(os.path.join(CAM_FOLDER, cameraFilename))
	camDataImg = camData['cam']
	camDataTs = camData['ts']

	vt, ct = getSynchArrays(inputTimestamps, camDataTs)
	numPoints = len(ct)

	newTs = camDataTs[0, :][ct].reshape(numPoints, 1)
	newCamDataImg = camDataImg[:,:,:,ct]
	newPredMat = predOrientationMatrices[:,:,vt]

	imgShape = camDataImg[:,:,:,0].shape
	nrows = imgShape[0]
	ncols = imgShape[1]

	finalImage = np.zeros((1000, 1600, 3), dtype=np.uint8)

	for t in range(numPoints):
		img = newCamDataImg[:,:,:,t]
		rmat = newPredMat[:,:,t]

		nic1 = np.transpose(np.indices((nrows,ncols)), (1, 2, 0))

		nic2 = np.zeros((nic1.shape[0], nic1.shape[1], 3))
		nic2[:,:,0] = 1
		nic2[:,:,1] = ((nrows/2.0) - nic1[:,:,0]) * ((math.pi/4)/nrows)
		nic2[:,:,2] = (nic1[:,:,1] - (ncols/2.0)) * ((math.pi/3)/ncols)

		nic3 = np.zeros((nic1.shape[0], nic1.shape[1], 3))
		nic3[:,:,0] = np.multiply(np.cos(nic2[:,:,1]), np.cos(nic2[:,:,2]))
		nic3[:,:,1] = np.multiply(np.cos(nic2[:,:,1]), np.sin(nic2[:,:,2]))
		nic3[:,:,2] = np.sin(nic2[:,:,1])

		nic3 = nic3.reshape(nic1.shape[0], nic1.shape[1], 3, 1)
		nic4 = np.matmul(rmat, nic3)
		nic4 = nic4.reshape(nic1.shape[0], nic1.shape[1], 3)

		nic5 = np.zeros((nic1.shape[0], nic1.shape[1], 3))
		nic5[:,:,0] = 1
		nic5[:,:,1] = np.arcsin(nic4[:,:,2])
		nic5[:,:,2] = np.arctan(np.divide(nic4[:,:,1], nic4[:,:,0]))
		# Global spherical coordinates calculated

		nic6 = np.zeros((nic1.shape[0], nic1.shape[1], 2))
		nic6[:,:,0] = (999/2.0) - (nic5[:,:,1] * 300)
		nic6[:,:,1] = (1599/2.0) + (nic5[:,:,2] * 450)
		nic6 = np.rint(nic6).astype(int)

		img = np.fliplr(img)
		finalImage[nic6[:,:,0], nic6[:,:,1], :] = img

		cv2.imshow('Image',finalImage)
		cv2.waitKey(10)
	cv2.destroyAllWindows()