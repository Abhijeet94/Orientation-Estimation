from scipy import io
import cv2, os
import math
import numpy as np
import random
import pdb

from constants import *
from plots import *
from utils import *

def quatAdd(a, b):
	return np.add(a, b)

def quatNorm(q):
	return np.linalg.norm(q)

def quatMultiply(a, b):
	a = a.reshape(4, 1)
	b = b.reshape(4, 1)

	afirst = a[0, 0]
	asecond = a[1:4, 0]

	bfirst = b[0, 0]
	bsecond = b[1:4, 0]

	resFirst = afirst * bfirst - np.dot(asecond, bsecond)
	resSecond = np.cross(asecond, bsecond) + afirst * bsecond + bfirst * asecond
	result = np.insert(resSecond, 0, resFirst).reshape(4, 1) 
	return result/quatNorm(result)

def quatConjugate(q):
	a = np.copy(q).reshape(4, 1)
	a[1, 0] = -1 * a[1, 0]
	a[2, 0] = -1 * a[2, 0]
	a[3, 0] = -1 * a[3, 0]
	return a

def quatInv(q):
	return quatConjugate(q) / (quatNorm(q) ** 2)

def qMul1(A, b):
	result = np.zeros((4, A.shape[1]))
	for i in range(A.shape[1]):
		result[:, i] = quatMultiply(A[:, i], b).reshape(4)
	return result

def qMul2(a, B):
	result = np.zeros((4, B.shape[1]))
	for i in range(B.shape[1]):
		result[:, i] = quatMultiply(a, B[:, i]).reshape(4)
	return result

def rotv2quat(r):
	rotNorm = np.linalg.norm(r, axis=0)
	angleCos = np.cos(rotNorm/2)
	angleSin = np.sin(rotNorm/2)

	isLow = rotNorm < 1e-20

	result = np.zeros((4, r.shape[1]))
	result[0, :] = angleCos
	result[1:4, isLow] = 0
	result[1:4, ~isLow] = np.multiply(np.divide(r[:, ~isLow], rotNorm[~isLow]), angleSin[~isLow])
	return result

def quat2rotv(q):
	modq = np.linalg.norm(q, axis=0)
	theta = np.arctan2(modq, q[0, :])

	isLow = theta < 1e-20

	result = np.zeros((3, q.shape[1]))
	result[:, isLow] = 0

	sin_alpha_w_by_2 = np.sin(theta/2)
	const = np.divide(theta[~isLow], sin_alpha_w_by_2[~isLow])
	result[:, ~isLow] = np.multiply(q[1:4, ~isLow], const)
	return result

def getRotatedG(Y):
	g = np.asarray([0, 0, 0, 1])
	result = np.zeros((3, Y.shape[1]))
	for i in range(Y.shape[1]):
		q_Yi_inverse = quatInv(Y[:, i])
		result[:, i] = (quatMultiply(quatMultiply(q_Yi_inverse, g), Y[:, i]))[1:4, 0]
	return result

def getQuatRotFromAngularVelocity(w, delta_t):
	# Assuming w is 3 X 1
	# Returns a 4 X 1 numpy array representing a Quaternion
	w = w.reshape(3, 1)
	rotNorm = np.linalg.norm(w)#math.sqrt(w[0, 0]**2 + w[1, 0]**2 + w[2, 0]**2)

	if rotNorm < 1e-20:
		return np.asarray([1, 0, 0, 0]).reshape(4, 1)

	angleCosine = math.cos(rotNorm * delta_t/2)
	angleSine = math.sin(rotNorm * delta_t/2)
	return np.asarray([angleCosine, angleSine * (w[0,0]/rotNorm), angleSine * (w[1,0]/rotNorm), angleSine * (w[2,0]/rotNorm)]).reshape(4, 1)

def calMeanQuat(Q):
	qcov = np.zeros((4, 4))
	for i in range(Q.shape[1]):
		q = Q[:, i].reshape(4, 1)
		qcov = qcov + np.matmul(q, np.transpose(q))
	qcov = qcov / (1.0 * Q.shape[1])

	w, v = np.linalg.eig(qcov)
	maxEigValIndex = np.argmax(w)
	return v[:, maxEigValIndex].reshape(4, 1)

def calMeanQuat2(Q, startValue = None):

	if startValue is None:
		startValue = np.asarray([0, 0, 1, 0]).reshape(4, 1)

	prevQBar = startValue
	listSize = Q.shape[1]
	e_quat = [None] * listSize
	e_vec = [None] * listSize
	numIterations = 0

	while True:
		e_mean = np.zeros((3, 1))
		for i in range(listSize):
			if np.linalg.norm(prevQBar) < 1e-30:
				e_quat[i] = Q[:, i]
			else:
				e_quat[i] = quatMultiply(Q[:, i], quatInv(prevQBar))
			e_vec[i] = quat2rotv(np.asarray(e_quat[i]))
			e_mean = e_mean + e_vec[i]
		e_mean = e_mean / (listSize * 1.0)

		if(np.linalg.norm(e_mean) < 1e-5):
			break

		e_mean_quat = rotv2quat(e_mean)
		e_mean_quat = e_mean_quat / quatNorm(e_mean_quat)
		prevQBar = quatMultiply(e_mean_quat, prevQBar)

		numIterations = numIterations + 1
		if numIterations > 50:
			break
	# print numIterations
	return prevQBar


def UKF(gyroData, accelerometerData, timestamps):

	# 6 X 6 
	positionCovParam = 0.05
	angularVelocityCovParam = 0.05
	Q_processNoiseCovariance = np.diag(np.concatenate((positionCovParam * np.ones(3), angularVelocityCovParam * np.ones(3))))

	# 6 X 6
	accCovParam = 0.7
	gyroCovParam = 0.05
	R_measurementNoiseCov = np.diag(np.concatenate((accCovParam * np.ones(3), gyroCovParam * np.ones(3))))

	# 6 X 6
	orientationCovParam = 0.1
	angVelCovParam = 0.01
	P_prevCovariance_P_km1 = np.diag(np.concatenate((orientationCovParam * np.ones(3), angVelCovParam * np.ones(3))))

	# 7 X 1
	prevStateEstimate_x_km1 = np.asarray([1, 0, 0, 0, 0, 0, 0]).reshape(7, 1)

	result = [None] * timestamps.size

	for index in range(timestamps.shape[1]):

		n = 6
		S = np.linalg.cholesky(np.add(P_prevCovariance_P_km1, Q_processNoiseCovariance))
		W = np.concatenate((math.sqrt(2*n) * S, -1 * math.sqrt(2*n) * S), axis = 1)

		X = np.zeros((7, 12))
		X[4:7, :] = W[3:6, :] + prevStateEstimate_x_km1[4:7, 0].reshape(3, 1)
		X[0:4, :] = qMul2(prevStateEstimate_x_km1[0:4, 0].reshape(4, 1), rotv2quat(W[0:3, :]))

		delta_t = 0.000001 if (index == 0) else timestamps[0, index] - timestamps[0, index - 1]
		q_delta = getQuatRotFromAngularVelocity(prevStateEstimate_x_km1[4:7, 0], delta_t)
		Y = np.zeros((7, 12))
		Y[4:7, :] = X[4:7, :]
		Y[0:4, :] = qMul1(X[0:4, :], q_delta)

		w_bar = np.mean(Y[4:7, :], axis=1)
		q_bar = calMeanQuat(Y[0:4, :])
		# q_bar = calMeanQuat2(Y[0:4, :], prevStateEstimate_x_km1[0:4, 0])

		W_script_prime = np.zeros((6, 12))
		W_script_prime[3:6, :] = Y[4:7, :] - w_bar.reshape(3, 1)
		W_script_prime[0:3, :] = quat2rotv(qMul1(Y[0:4, :], quatInv(q_bar)))

		Ycov_P_k_bar = np.matmul(W_script_prime, np.transpose(W_script_prime)) / 12.0

		Z = np.zeros((6, 12))
		Z[3:6, :] = Y[4:7, :]
		Z[0:3, :] = getRotatedG(Y[0:4, :])

		z_k_bar = np.mean(Z, axis=1).reshape(6, 1)

		P_zz = np.matmul((Z - z_k_bar), np.transpose((Z - z_k_bar))) / 12.0

		P_vv = np.add(P_zz, R_measurementNoiseCov)

		P_xz = np.matmul(W_script_prime, np.transpose((Z - z_k_bar))) / 12.0

		kalmanGain_K_k = np.matmul(P_xz, np.linalg.inv(P_vv))

		actualMeasurement = np.asarray([accelerometerData[index, 0], accelerometerData[index, 1], accelerometerData[index, 2], gyroData[index, 0], gyroData[index, 1], gyroData[index, 2]]).reshape(6, 1)
		innovation_v_k = np.subtract(actualMeasurement, z_k_bar)
		if np.linalg.norm(accelerometerData[index, :]) < 0.9 or np.linalg.norm(accelerometerData[index, :]) > 1.1:
			innovation_v_k[0:3, 0] = 0
		updateVal = np.matmul(kalmanGain_K_k, innovation_v_k)

		newStateEstimate_x_k = np.zeros((7, 1))
		newStateEstimate_x_k[4:7, 0] = w_bar + updateVal[3:6, 0]
		newStateEstimate_x_k[0:4, 0] = quatMultiply(q_bar, rotv2quat(updateVal[0:3, 0].reshape(3, 1))).reshape(4)

		newCovariance_P_k = Ycov_P_k_bar - np.matmul(np.matmul(kalmanGain_K_k, P_vv), np.transpose(kalmanGain_K_k))

		result[index] = newStateEstimate_x_k
		prevStateEstimate_x_km1 = newStateEstimate_x_k
		P_prevCovariance_P_km1 = newCovariance_P_k

	return result