from scipy import io
import cv2, os
import math
import numpy as np

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
	return np.insert(resSecond, 0, resFirst).reshape(4, 1)

def quatConjugate(q):
	a = np.copy(q).reshape(4, 1)
	a[1, 0] = -1 * a[1, 0]
	a[2, 0] = -1 * a[2, 0]
	a[3, 0] = -1 * a[3, 0]
	return a

def quatNorm(q):
	q = q.reshape(4, 1)
	return math.sqrt(q[0, 0] ** 2 + q[1, 0] ** 2 + q[2, 0] ** 2 + q[3, 0] ** 2)

def quatInv(q):
	return quatConjugate(q) / (quatNorm(q) ** 2)

def getNormOfVector(x):
	x = x.reshape(x.size, 1)
	s = 0.0
	for i in range(x.size):
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
	angleCos = math.cos(rotNorm/2)
	angleSin = math.sin(rotNorm/2)
	return np.asarray([angleCos, angleSin * (rot[0,0]/rotNorm), angleSin * (rot[1,0]/rotNorm), angleSin * (rot[2,0]/rotNorm)]).reshape(4, 1)

def getRotationVectorFromQuat(q):
	# q is 4 X 1 quaternion
	# return 3 X 1 rotation vector
	q = q.reshape(4, 1)
	mod_w = math.sqrt(q[1, 0]**2 + q[2, 0]**2 + q[3, 0]**2)
	theta = 2 * math.atan2(mod_w, q[0,0])
	if abs(theta - 0) < 0.00001:
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
	angleCosine = math.cos(rotNorm * delta_t/2)
	angleSine = math.sin(rotNorm * delta_t/2)
	return np.asarray([angleCosine, angleSine * (w[0,0]/rotNorm), angleSine * (w[1,0]/rotNorm), angleSine * (w[2,0]/rotNorm)])

def calMeanOfAngularVelocity(Y):
	# Assuming Y is 7 X 1
	# Angular velocity is in the last three components
	result = np.zeros((3, 1))
	for y in Y:
		result = result + y[4:7, 0].reshape(3, 1)
	return result/len(Y)

def calMeanOfQuat(Y, startValue = None):
	# Assuming Y is list of np arrays of size 7 X 1
	# Quat is in the first four components
	# result is 4 X 1, [4 X 1]
	# first representing the mean, the other last iteration e_i
	# print Y

	if startValue is None:
		startValue = np.asarray([0, 0, 0, 0]).reshape(4, 1)

	prevQBar = startValue
	n = 6
	e_quat = [None] * (2 * n)
	e_vec = [None] * (2 * n)

	while True:
		# print 'looping for quat averaging!'
		e_mean = np.zeros((3, 1))
		for i in range(2 * n):
			if abs(getNormOfVector(prevQBar) - 0) < 0.001:
				e_quat[i] = Y[i][0:4, 0]
			else:
				e_quat[i] = quatMultiply(Y[i][0:4, 0], quatInv(prevQBar))
			e_vec[i] = getRotationVectorFromQuat(e_quat[i])
			e_mean = e_mean + e_vec[i]
		e_mean = e_mean / (2 * n * 1.0)
		# print 'in cal mean'
		# print e_quat, e_mean
		if abs(getNormOfVector(e_mean) - 0) < 0.001:
			e_mean_quat = prevQBar
		else:
			e_mean_quat = getQuatFromRotationVector(e_mean)
		prevQBar = quatMultiply(e_mean_quat, prevQBar)

		if(math.sqrt(e_mean[0, 0] ** 2 + e_mean[1, 0] ** 2 + e_mean[2, 0] ** 2) < 0.05):
			break

	return prevQBar, e_quat

def getAccelQuatFromAccelData(x, y, z):
	# return 4 X 1 numpy array
	return np.asarray([0, x, y, z])

def UKF(gyroData, accelerometerData, timestamps):
	'''
	gyroData - wx, wy, wz - angular velocities, wrt frame - (body)?
	accelerometerData - ax, ay, az - accelerations along the correspondig axes - wrt frame - (body)?
	timestamp - sequence of time at which the above data is measured

	UKF() - calculate the orientation of the body, using the Unscented Kalman Filter
	'''

	positionCovParam = 4
	angularVelocityCovParam = 5
	# 6 X 6 
	processNoiseCovariance_Q = np.diag(np.concatenate((positionCovParam * np.ones(3), angularVelocityCovParam * np.ones(3))))

	accCovParam = 4
	gyroCovParam = 5
	# 6 X 6
	gyroNoiseCovariance_R = np.diag(np.concatenate((accCovParam * np.ones(3), gyroCovParam * np.ones(3))))
	accelerometerNoiseCovariance_R = gyroNoiseCovariance_R

	# 7 X 1
	prevStateEstimate_x_km1 = np.zeros(7).reshape(7, 1)

	# 6 X 6
	prevCovariance_P_km1 = np.ones((6, 6))

	# Final result - Orientation represented in quaternions
	result = [None] * timestamps.size

	for index in range(timestamps.shape[1]):
		for mindex, measurement in enumerate(['gyro', 'accelerometer']):
			# print 'Iteration: ' + str(index)
			# Computing Sigma Points
			# Cholesky decomposition
			n = 6
			S = np.linalg.cholesky(np.add(prevCovariance_P_km1, processNoiseCovariance_Q))
			S_hsplit = np.hsplit(S, n)
			# 2n points - n X 1
			W = [None] * (2 * n)
			for i in range(n):
				W[i] = math.sqrt(2 * n) * S_hsplit[i]
				W[i + n] = -1 * math.sqrt(2 * n) * S_hsplit[i]
			# print W
			# exit()

			# 2n points - 7 X 1
			X = [None] * (2 * n)
			q_prevState, w_prevState = getOriQuatAngvelFromState(prevStateEstimate_x_km1)
			for i in range(2 * n):
				qv_W_i, w_W_i = getOriQuatAngvelFromVector(W[i])
				if abs(getNormOfVector(qv_W_i) - 0) < 0.001:
					quatPart = q_prevState
				else:
					quatPart = quatMultiply(q_prevState, getQuatFromRotationVector(qv_W_i))
				angVelPart = np.add(w_prevState, w_W_i)
				X[i] = getStateFromOriQuatAngvel(quatPart, angVelPart)
			# print X
			# exit()
			
			# 2n points - 7 X 1
			Y = [None] * (2 * n)
			delta_t = 0.000001 if (index == 0) else timestamps[0, index] - timestamps[0, index - 1]
			if abs(getNormOfVector(w_prevState) - 0) < 0.001:
				q_delta = np.asarray([1, 0, 0, 0]).reshape(4, 1)
			else:
				q_delta = getQuatRotFromAngularVelocity(w_prevState, delta_t)

			for i in range(2 * n):
				q_Xi, w_Xi = getOriQuatAngvelFromState(X[i])
				quatPart = quatMultiply(q_Xi, q_delta)
				Y[i] = getStateFromOriQuatAngvel(quatPart, w_prevState)
			# print Y
			# exit()

			# 3 X 1
			w_bar = calMeanOfAngularVelocity(Y)
			# 4 X 1, 4 X 1
			q_bar, q_lastIter_e = calMeanOfQuat(Y, q_prevState)
			# 7 X 1
			Ymean_x_k_bar = getStateFromOriQuatAngvel(q_bar, w_bar)

			# 2n points - 6 X 1
			W_script_prime = [None] * (2 * n)
			for i in range(2 * n):
				q_Yi, w_Yi = getOriQuatAngvelFromState(Y[i])
				quatPart = q_lastIter_e[i].reshape(4, 1)
				angVelPart = w_Yi.reshape(3, 1) - w_bar
				W_script_prime[i] = getStateFromOriQuatAngvel(getRotationVectorFromQuat(quatPart), angVelPart)
				# print W_script_prime[i].shape

			# 6 X 6
			Ycov_P_k_bar = np.zeros((6, 6))
			for i in range(2 * n):
				Ycov_P_k_bar = Ycov_P_k_bar + np.matmul(W_script_prime[i], np.transpose(W_script_prime[i]))
			Ycov_P_k_bar = Ycov_P_k_bar / (2 * n)

			##########################
			### Measurement update ###
			##########################

			# 2n points - 6 X 1
			Z = [None] * (2 * n)
			if measurement == 'gyro':
				for i in range(2 * n):
					q_Yi, w_Yi = getOriQuatAngvelFromState(Y[i])
					quatPart = np.asarray([0, 0, 0]).reshape(3, 1)
					angVelPart = w_Yi
					Z[i] = getStateFromOriQuatAngvel(quatPart, angVelPart)
			elif measurement == 'accelerometer':
				for i in range(2 * n):
					q_Yi, w_Yi = getOriQuatAngvelFromState(Y[i])
					g = np.asarray([0, 0, 0, 9.8])
					if abs(getNormOfVector(q_Yi) - 0) < 0.001:
						q_Yi_inverse = np.asarray([1, 0, 0, 0]).reshape(4, 1)
					else:
						q_Yi_inverse = quatInv(q_Yi)
					quatPart = getRotationVectorFromQuat(quatMultiply(quatMultiply(q_Yi_inverse, g), q_Yi))
					angVelPart = np.asarray([0, 0, 0]).reshape(3, 1)
					Z[i] = getStateFromOriQuatAngvel(quatPart, angVelPart)

			# 6 X 1
			z_k_bar = np.zeros((6, 1))
			for i in range(2 * n):
				z_k_bar = z_k_bar + Z[i]
			z_k_bar = z_k_bar / (2 * n)

			# 6 X 6
			P_zz = np.zeros((6, 6))
			for i in range(2 * n):
				P_zz = P_zz + np.matmul((Z[i] - z_k_bar), np.transpose((Z[i] - z_k_bar)))
			P_zz = P_zz / (2 * n)

			# 6 X 6
			P_vv = np.add(P_zz, gyroNoiseCovariance_R)

			# 6 X 6
			P_xz = np.zeros((6, 6))
			for i in range(2 * n):
				P_xz = P_xz + np.matmul(W_script_prime[i], np.transpose((Z[i] - z_k_bar)))
			P_xz = P_xz / (2 * n)

			# 6 X 6
			kalmanGain_K_k = np.matmul(P_xz, np.linalg.inv(P_vv))

			if measurement == 'gyro':
				# 6 X 1
				actualMeasurement = np.asarray([0, 0, 0, gyroData[index, 0], gyroData[index, 1], gyroData[index, 2]]).reshape(6, 1)
				innovation_v_k = np.subtract(actualMeasurement, z_k_bar)
			elif measurement == 'accelerometer':
				accelQuat = getAccelQuatFromAccelData(accelerometerData[index, 0], accelerometerData[index, 1], accelerometerData[index, 2])
				rotVecForAccQuat = getRotationVectorFromQuat(accelQuat)
				actualMeasurement = np.asarray([rotVecForAccQuat[0, 0], rotVecForAccQuat[1, 0], rotVecForAccQuat[2, 0], 0, 0, 0]).reshape(6, 1)
				# 6 X 1
				innovation_v_k = np.subtract(actualMeasurement, z_k_bar)

			updateVal = np.matmul(kalmanGain_K_k, innovation_v_k)
			qv_updateVal, w_updateVal = getOriQuatAngvelFromVector(updateVal)
			q_Ymean_x_k_bar, w_Ymean_x_k_bar = getOriQuatAngvelFromState(Ymean_x_k_bar)
			if abs(getNormOfVector(qv_updateVal) - 0) < 0.001:
				quatPart = q_Ymean_x_k_bar
			else:
				quatPart = quatMultiply(q_Ymean_x_k_bar, getQuatFromRotationVector(qv_updateVal))
			angVelPart = np.add(w_Ymean_x_k_bar, w_updateVal)
			# 7 X 1
			newStateEstimate_x_k = getStateFromOriQuatAngvel(quatPart, angVelPart)

			# 6 X 6
			newCovariance_P_k = Ycov_P_k_bar + (-1) * (np.matmul(np.matmul(kalmanGain_K_k, P_vv), np.transpose(kalmanGain_K_k)))

			### 
			### Store Values
			###

			result[index] = newStateEstimate_x_k
			prevStateEstimate_x_km1 = newStateEstimate_x_k
			prevCovariance_P_km1 = newCovariance_P_k

	return result