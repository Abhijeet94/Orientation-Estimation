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
	return result#/quatNorm(result)

def quatConjugate(q):
	a = np.copy(q).reshape(4, 1)
	a[1, 0] = -1 * a[1, 0]
	a[2, 0] = -1 * a[2, 0]
	a[3, 0] = -1 * a[3, 0]
	return a

def quatNorm(q):
	# print q
	q = q.reshape(4, 1)
	return math.sqrt(q[0, 0] ** 2 + q[1, 0] ** 2 + q[2, 0] ** 2 + q[3, 0] ** 2)

def quatInv(q):
	return quatConjugate(q) / (quatNorm(q) ** 2)

def getNormOfVector(x):
	x = x.reshape(x.size, 1)
	s = 0.0
	for i in range(x.size):
		if abs(x[i, 0]) < 1e-20:
			s = s + 0
		else:
			s = s + x[i, 0] ** 2
	return math.sqrt(s)

def getOriQuatAngvelFromState(state):
	# Assuming state is a 7 X 1 vector
	quat = state[0:4, 0]
	angVel = state[4:7, 0]
	return quat, angVel

def getOriQuatAngvelFromVector(vec):
	# Assuming vec is a n X 1 vector, with n>3
	n = vec.shape[0]
	quat = vec[0: (n-3), 0]
	angVel = vec[(n-3) : n, 0]
	return quat, angVel

def getStateFromOriQuatAngvel(q, w):
	# Concatenating into single numpy array
	# For example, if q is 4 X 1 and w is 3 X 1, result is 7 X 1
	return np.concatenate((q.reshape(q.size, 1), w.reshape(w.size, 1)))

def getQuatFromRotationVector(rot):
	# Assuming rot is 3 X 1
	# Returns a 4 X 1 numpy array representing a Quaternion
	# print rot
	rot = rot.reshape(3, 1)
	rotNorm = math.sqrt(rot[0, 0]**2 + rot[1, 0]**2 + rot[2, 0]**2)
	if abs(rotNorm - 0) < 1e-20:
		return np.asarray([1, 0, 0, 0]).reshape(4, 1)
	angleCos = math.cos(rotNorm/2)
	angleSin = math.sin(rotNorm/2)
	return np.asarray([angleCos, angleSin * (rot[0,0]/rotNorm), angleSin * (rot[1,0]/rotNorm), angleSin * (rot[2,0]/rotNorm)]).reshape(4, 1)

def getRotationVectorFromQuat(q):
	# q is 4 X 1 quaternion
	# return 3 X 1 rotation vector
	q = q.reshape(4, 1)
	mod_w = math.sqrt(q[1, 0]**2 + q[2, 0]**2 + q[3, 0]**2)
	theta = 2 * math.atan2(mod_w, q[0,0])
	if abs(theta - 0) < 1e-20:
		return np.asarray([0, 0, 0]).reshape(3, 1)
	else:
		sin_alpha_w_by_2 = math.sin(theta/2)
		const = theta / sin_alpha_w_by_2
		return np.asarray([q[1,0] * const, q[2,0] * const, q[3,0] * const]).reshape(3, 1)

def getQuatRotFromAngularVelocity(w, delta_t):
	# Assuming w is 3 X 1
	# Returns a 4 X 1 numpy array representing a Quaternion
	w = w.reshape(3, 1)
	rotNorm = math.sqrt(w[0, 0]**2 + w[1, 0]**2 + w[2, 0]**2)

	if rotNorm < 1e-20:
		return np.asarray([1, 0, 0, 0]).reshape(4, 1)

	angleCosine = math.cos(rotNorm * delta_t/2)
	angleSine = math.sin(rotNorm * delta_t/2)
	return np.asarray([angleCosine, angleSine * (w[0,0]/rotNorm), angleSine * (w[1,0]/rotNorm), angleSine * (w[2,0]/rotNorm)]).reshape(4, 1)

def calMeanOfAngularVelocity(Y):
	# Assuming Y is 7 X 1
	# Angular velocity is in the last three components
	result = np.zeros((3, 1))
	for y in Y:
		# print y[4:7, 0].reshape(3, 1)
		result = result + y[4:7, 0].reshape(3, 1)
	return result/len(Y)

def remAccUpdateIfWrongAccel(acceleration, inn):
	acc = math.sqrt(acceleration[0, 0]**2 + acceleration[1, 0]**2 + acceleration[2, 0]**2)
	if acc < 0.75 or acc > 1.25:
		return np.asarray([0,0,0,inn[3,0],inn[4,0],inn[5,0]]).reshape(6, 1)
	else:
		return inn

def calMeanOfQuat(Y, startValue = None):
	# Assuming Y is list of np arrays of size 7 X 1
	# Quat is in the first four components
	# result is 4 X 1, [4 X 1]
	# first representing the mean, the other last iteration e_i

	if startValue is None:
		startValue = np.asarray([0, 0, 1, 0]).reshape(4, 1)

	prevQBar = startValue
	listSize = len(Y)
	e_quat = [None] * listSize
	e_vec = [None] * listSize
	numIterations = 0

	while True:
		e_mean = np.zeros((3, 1))
		for i in range(listSize):
			if getNormOfVector(prevQBar) < 1e-30:
				e_quat[i] = Y[i][0:4, 0]
			else:
				e_quat[i] = quatMultiply(Y[i][0:4, 0], quatInv(prevQBar))
			e_vec[i] = getRotationVectorFromQuat(e_quat[i])
			e_mean = e_mean + e_vec[i]
		e_mean = e_mean / (listSize * 1.0)

		if(math.sqrt(e_mean[0, 0] ** 2 + e_mean[1, 0] ** 2 + e_mean[2, 0] ** 2) < 1e-5):
			break

		e_mean_quat = getQuatFromRotationVector(e_mean)
		e_mean_quat = e_mean_quat / quatNorm(e_mean_quat)
		prevQBar = quatMultiply(e_mean_quat, prevQBar)

		numIterations = numIterations + 1
		if numIterations > 20:
			break
	# print numIterations
	return prevQBar, e_quat

def calMeanOfQuat2(Y):
	qcov = np.zeros((4, 4))
	for i in range(len(Y)):
		q = Y[i][0:4, 0].reshape(4, 1)
		# print q
		qcov = qcov + np.matmul(q, np.transpose(q))
	qcov = qcov / (1.0 * len(Y))
	w, v = np.linalg.eig(qcov)
	maxEigValIndex = np.argmax(w)
	# print v[:, maxEigValIndex].reshape(4, 1)
	return v[:, maxEigValIndex].reshape(4, 1)

def testQuatAveraging():
	Y = []

	# Y.append(np.asarray([0, 0.707, 0.5, 0.5]).reshape(4, 1))
	# Y.append(np.asarray([0, 0.707, 0.5, 0.5]).reshape(4, 1))
	# Y.append(np.asarray([0, 0.707, 0.5, 0.5]).reshape(4, 1))

	Y.append(np.asarray([0, 0, 1, 0]).reshape(4, 1))
	Y.append(np.asarray([0, 0, 1, 0]).reshape(4, 1))
	Y.append(np.asarray([0, 0, 1, 0]).reshape(4, 1))

	# Y.append(np.asarray([0, 0.707, 0.707, 0]).reshape(4, 1))
	# Y.append(np.asarray([0, 0.707, 0.707, 0]).reshape(4, 1))
	# Y.append(np.asarray([0, 0.707, 0.707, 0]).reshape(4, 1))

	res1, res2 = calMeanOfQuat(Y)
	return res1

def testQuatToRotToQuat():
	# x = np.asarray([0, 1, 0, 0]).reshape(4, 1)
	x = np.asarray([0, 0.707, 0.5, 0.5]).reshape(4, 1)
	# x = np.asarray([1, 0, 0, 0]).reshape(4, 1)
	y = getRotationVectorFromQuat(x)
	z = getQuatFromRotationVector(y)
	print x
	print z/quatNorm(z)


def calMean_z_k(Z):
	# 6 X 1
	ZZ_w_bar = np.zeros((3, 1))
	ZZ = [None] * len(Z)
	for i in range(len(Z)):
		ZZ[i] = np.zeros((7, 1))
		qv_ZZ_i, w_ZZ_i = getOriQuatAngvelFromVector(Z[i])
		ZZ[i][0:4, 0] = getQuatFromRotationVector(qv_ZZ_i).reshape(4)
		ZZ[i][4:7, 0] = w_ZZ_i
		ZZ_w_bar = ZZ_w_bar + w_ZZ_i.reshape(3, 1)
	ZZ_w_bar = ZZ_w_bar / len(Z)

	# ZZ_q_bar, throwaway = calMeanOfQuat(ZZ)
	ZZ_q_bar = calMeanOfQuat2(ZZ)
	return getStateFromOriQuatAngvel(getRotationVectorFromQuat(ZZ_q_bar), ZZ_w_bar)

def normalizeQuatInStateVector(X):
	norm = quatNorm(X[0:4, 0])
	X[0, 0] = X[0, 0] / norm
	X[1, 0] = X[1, 0] / norm
	X[2, 0] = X[2, 0] / norm
	X[3, 0] = X[3, 0] / norm
	return X

def checkGyroIntegration(gyroData, timestamps):
	result = [None] * timestamps.shape[1]
	prevQ = np.asarray([1, 0, 0, 0])

	for index in range(timestamps.shape[1]):
		delta_t = 0.000001 if (index == 0) else timestamps[0, index] - timestamps[0, index - 1]
		wdt_x = gyroData[index, 0] * delta_t
		wdt_y = gyroData[index, 1] * delta_t
		wdt_z = gyroData[index, 2] * delta_t

		result[index] = quatMultiply(prevQ, getQuatFromRotationVector(np.asarray([wdt_x, wdt_y, wdt_z]).reshape(3, 1)))
		prevQ = result[index]
	return result

def UKF(gyroData, accelerometerData, timestamps):
	'''
	gyroData - wx, wy, wz - angular velocities, wrt frame - (body)?
	accelerometerData - ax, ay, az - accelerations along the correspondig axes - wrt frame - (body)?
	timestamp - sequence of time at which the above data is measured

	UKF() - calculate the orientation of the body, using the Unscented Kalman Filter
	'''

	# 6 X 6 
	positionCovParam = 500
	angularVelocityCovParam = 0.01
	Q_processNoiseCovariance = np.diag(np.concatenate((positionCovParam * np.ones(3), angularVelocityCovParam * np.ones(3))))

	# 6 X 6
	accCovParam = 0.0500
	gyroCovParam = 0.01
	R_measurementNoiseCov = np.diag(np.concatenate((accCovParam * np.ones(3), gyroCovParam * np.ones(3))))

	# 6 X 6
	orientationCovParam = 0.1
	angVelCovParam = 0.01
	P_prevCovariance_P_km1 = np.diag(np.concatenate((orientationCovParam * np.ones(3), angVelCovParam * np.ones(3))))

	# 7 X 1
	prevStateEstimate_x_km1 = np.asarray([1, 0, 0, 0, 0, 0, 0]).reshape(7, 1)

	# Final result - Orientation represented in quaternions
	result = [None] * timestamps.size

	for index in range(timestamps.shape[1]):

		###############################################################################################
		###############################################################################################

		# r, p, y = transforms3d.euler.quat2euler(prevStateEstimate_x_km1[0:4, 0], 'sxyz')
		# print r, p, y

		# pdb.set_trace()

		# Computing Sigma Points
		# Cholesky decomposition
		n = 6
		S = np.linalg.cholesky(np.add(P_prevCovariance_P_km1, Q_processNoiseCovariance))

		S_hsplit = np.hsplit(S, n)

		# 2n points - n X 1
		W = [None] * (2 * n)
		for i in range(n):
			W[i] = math.sqrt(2 * n) * S_hsplit[i]
			W[i + n] = -1 * math.sqrt(2 * n) * S_hsplit[i]
		# print "W" + str(W)

		###############################################################################################
		###############################################################################################

		# 2n points - 7 X 1
		X = [None] * (2 * n)
		q_prevState, w_prevState = getOriQuatAngvelFromState(prevStateEstimate_x_km1)
		for i in range(2 * n):
			qv_W_i, w_W_i = getOriQuatAngvelFromVector(W[i])
			quatPart = quatMultiply(q_prevState, getQuatFromRotationVector(qv_W_i))
			angVelPart = np.add(w_prevState, w_W_i)
			X[i] = getStateFromOriQuatAngvel(quatPart, angVelPart)
		# print "X" + str(X)
		
		###############################################################################################
		###############################################################################################

		# 2n points - 7 X 1
		Y = [None] * (2 * n)
		delta_t = 0.000001 if (index == 0) else timestamps[0, index] - timestamps[0, index - 1]
		q_delta = getQuatRotFromAngularVelocity(w_prevState, delta_t)

		for i in range(2 * n):
			q_Xi, w_Xi = getOriQuatAngvelFromState(X[i])
			quatPart = quatMultiply(q_Xi, q_delta)
			Y[i] = getStateFromOriQuatAngvel(quatPart, w_Xi)
		# print "Y" + str(Y)

		###############################################################################################
		###############################################################################################

		# 3 X 1
		w_bar = calMeanOfAngularVelocity(Y)
		# 4 X 1, 4 X 1
		# q_bar, q_lastIter_e = calMeanOfQuat(Y, q_prevState)
		q_bar = calMeanOfQuat2(Y)
		q_bar = q_bar / quatNorm(q_bar)
		# print 'qbar - ' + str(q_bar)
		# 7 X 1
		Ymean_x_k_bar = getStateFromOriQuatAngvel(q_bar, w_bar)
		# print "Ymean_x_k_bar" + str(Ymean_x_k_bar)

		###############################################################################################
		###############################################################################################

		# 2n points - 6 X 1
		W_script_prime = [None] * (2 * n)
		for i in range(2 * n):
			q_Yi, w_Yi = getOriQuatAngvelFromState(Y[i])
			# quatPart = q_lastIter_e[i].reshape(4, 1)
			quatPart = quatMultiply(q_Yi, quatInv(q_bar)).reshape(4, 1)
			angVelPart = w_Yi.reshape(3, 1) - w_bar
			W_script_prime[i] = getStateFromOriQuatAngvel(getRotationVectorFromQuat(quatPart), angVelPart)

		###############################################################################################
		###############################################################################################

		# 6 X 6
		Ycov_P_k_bar = np.zeros((6, 6))
		for i in range(2 * n):
			Ycov_P_k_bar = Ycov_P_k_bar + np.matmul(W_script_prime[i], np.transpose(W_script_prime[i]))
		Ycov_P_k_bar = Ycov_P_k_bar / (2 * n)
		# print "Ycov_P_k_bar" + str(Ycov_P_k_bar)

		# newStateEstimate_x_k = Ymean_x_k_bar 			# testing code !! remove later - testing with prediction only !!
		# newCovariance_P_k = Ycov_P_k_bar 				# testing code !! remove later - testing with prediction only !!
		# newStateEstimate_x_k[4, 0] = gyroData[index, 0] # testing code !! remove later - testing with prediction only !!
		# newStateEstimate_x_k[5, 0] = gyroData[index, 1] # testing code !! remove later - testing with prediction only !!
		# newStateEstimate_x_k[6, 0] = gyroData[index, 2] # testing code !! remove later - testing with prediction only !!
		# print gyroData[index, :]

		##########################
		### Measurement update ###
		##########################

		# 2n points - 6 X 1
		Z = [None] * (2 * n)
		for i in range(2 * n):
			q_Yi, w_Yi = getOriQuatAngvelFromState(Y[i])
			g = np.asarray([0, 0, 0, 1])
			q_Yi_inverse = quatInv(q_Yi)
			gDir = (quatMultiply(quatMultiply(q_Yi_inverse, g), q_Yi))[1:4, 0]
			# if getNormOfVector(gDir) > 1e-10:
			# 	gDir = gDir/getNormOfVector(gDir)
			angVelPart = w_Yi
			Z[i] = getStateFromOriQuatAngvel(gDir, angVelPart)
			# print index
			# print Z[i][2]
			# pdb.set_trace()

		###############################################################################################
		###############################################################################################

		# 6 X 1
		z_k_bar = np.zeros((6, 1))
		for i in range(2 * n):
			z_k_bar = z_k_bar + Z[i]
		z_k_bar = z_k_bar / (2 * n)
		# z_k_bar = calMean_z_k(Z)

		# 6 X 6
		P_zz = np.zeros((6, 6))
		for i in range(2 * n):
			P_zz = P_zz + np.matmul((Z[i] - z_k_bar), np.transpose((Z[i] - z_k_bar)))
		P_zz = P_zz / (2 * n)
		# if index>1000 and index<1500:
		# 	print P_zz

		# 6 X 6
		P_vv = np.add(P_zz, R_measurementNoiseCov)

		# 6 X 6
		P_xz = np.zeros((6, 6))
		for i in range(2 * n):
			P_xz = P_xz + np.matmul(W_script_prime[i], np.transpose((Z[i] - z_k_bar)))
		P_xz = P_xz / (2 * n)

		###############################################################################################
		###############################################################################################

		# 6 X 6
		kalmanGain_K_k = np.matmul(P_xz, np.linalg.inv(P_vv))
		# print kalmanGain_K_k

		# 6 X 1
		accelMeas = np.asarray([accelerometerData[index, 0], accelerometerData[index, 1], accelerometerData[index, 2]]).reshape(3, 1)
		# print getNormOfVector(accelMeas)
		actualMeasurement = np.asarray([accelMeas[0, 0], accelMeas[1, 0], accelMeas[2, 0], gyroData[index, 0], gyroData[index, 1], gyroData[index, 2]]).reshape(6, 1)
		innovation_v_k = np.subtract(actualMeasurement, z_k_bar)
		# print actualMeasurement, z_k_bar
		# print innovation_v_k
		# innovation_v_k = remAccUpdateIfWrongAccel(accelMeas, innovation_v_k)
		# innovation_v_k[3:6, 0] = 0
		# innovation_v_k[0:3, 0] = 0

		updateVal = np.matmul(kalmanGain_K_k, innovation_v_k)
		qv_updateVal, w_updateVal = getOriQuatAngvelFromVector(updateVal)
		q_Ymean_x_k_bar, w_Ymean_x_k_bar = getOriQuatAngvelFromState(Ymean_x_k_bar)
		quatPart = quatMultiply(q_Ymean_x_k_bar, getQuatFromRotationVector(qv_updateVal))
		angVelPart = np.add(w_Ymean_x_k_bar, w_updateVal)
		# 7 X 1
		newStateEstimate_x_k = getStateFromOriQuatAngvel(quatPart, angVelPart)

		# 6 X 6
		newCovariance_P_k = Ycov_P_k_bar - np.matmul(np.matmul(kalmanGain_K_k, P_vv), np.transpose(kalmanGain_K_k))
		# print newCovariance_P_k

		### 
		### Store Values
		###

		result[index] = newStateEstimate_x_k
		prevStateEstimate_x_km1 = newStateEstimate_x_k
		P_prevCovariance_P_km1 = newCovariance_P_k

		# print np.linalg.norm(P_prevCovariance_P_km1)

	return result