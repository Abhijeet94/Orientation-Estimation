import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from time import sleep

from constants import *
from utils import *

fig = plt.figure()

def rotplot(R,currentAxes=None):
	# This is a simple function to plot the orientation
	# of a 3x3 rotation matrix R in 3-D
	# You should modify it as you wish for the project.

	lx = 3.0
	ly = 1.5
	lz = 1.0

	x = .5*np.array([[+lx, -lx, +lx, -lx, +lx, -lx, +lx, -lx],
		[+ly, +ly, -ly, -ly, +ly, +ly, -ly, -ly],
		[+lz, +lz, +lz, +lz, -lz, -lz, -lz, -lz]])

	xp = np.dot(R,x);
	ifront = np.array([0, 2, 6, 4, 0])
	iback = np.array([1, 3, 7, 5, 1])
	itop = np.array([0, 1, 3, 2, 0])
	ibottom = np.array([4, 5, 7, 6, 4])
	
	if currentAxes:
		ax = currentAxes;
	else:
		# fig = plt.figure()
		ax = fig.gca(projection='3d')

	ax.plot(xp[0,itop], xp[1,itop], xp[2,itop], 'k-')
	ax.plot(xp[0,ibottom], xp[1,ibottom], xp[2,ibottom], 'k-')
	
	rectangleFront = a3.art3d.Poly3DCollection([list(zip(xp[0,ifront], xp[1,ifront],xp[2,ifront]))])
	rectangleFront.set_facecolor('Blue')
	ax.add_collection(rectangleFront)
	
	rectangleBack = a3.art3d.Poly3DCollection([list(zip(xp[0,iback], xp[1,iback],xp[2,iback]))])
	rectangleBack.set_facecolor('Red')
	ax.add_collection(rectangleBack)

	ax.set_aspect('equal')
	ax.set_xlim3d(-2, 2)
	ax.set_ylim3d(-2, 2)
	ax.set_zlim3d(-2, 2)
	
	return ax


def sample():
	# Example usage: Putting two rotations on one graph.
	REye = np.eye(3)
	myAxis = rotplot(REye)
	RTurn = np.array([[np.cos(np.pi/2),0,np.sin(np.pi/2)],[0,1,0],[-np.sin(np.pi/2),0,np.cos(np.pi/2)]])
	rotplot(RTurn,myAxis)
	plt.show()

def viewVicon():
	data = loadFile(os.path.join(VICON_FOLDER, 'viconRot3.mat'))
	data = data['rots']
	numInstances = data.shape[2]

	# plt.show()
	plt.ion()

	myAxis = None
	for i in range(numInstances/2, numInstances):
		plt.clf()
		myAxis = rotplot(data[:, :, i])
		fig.canvas.draw()
		fig.canvas.flush_events()
		fig.show()
		# print transforms3d.euler.mat2euler(data[:, :, i], 'sxyz')

def plotEulerAnglesVicon():
	data = loadFile(os.path.join(VICON_FOLDER, 'viconRot5.mat'))
	ts = data['ts']
	data = data['rots']
	numInstances = data.shape[2]

	roll_pitch_yaw = np.zeros((3, numInstances))

	for i in range(numInstances):
		r, p, y = transforms3d.euler.mat2euler(data[:, :, i], 'sxyz')
		roll_pitch_yaw[0, i] = r
		roll_pitch_yaw[1, i] = p
		roll_pitch_yaw[2, i] = y

	plt.subplot(311)
	plt.plot(ts.reshape(numInstances, 1), roll_pitch_yaw[0, :].reshape(numInstances, 1), 'r-')

	plt.subplot(312)
	plt.plot(ts.reshape(numInstances, 1), roll_pitch_yaw[1, :].reshape(numInstances, 1), 'b-')

	plt.subplot(313)
	plt.plot(ts.reshape(numInstances, 1), roll_pitch_yaw[2, :].reshape(numInstances, 1), 'g-')

	plt.show()

def plotGTruthAndPredictions(viconFile, predictions, predTimestamps):
	viconData = loadFile(os.path.join(VICON_FOLDER, viconFile))
	viconTs = viconData['ts']
	viconMatrices = viconData['rots']
	numInstances = viconMatrices.shape[2]
	gt_roll_pitch_yaw = np.zeros((3, numInstances))

	for i in range(numInstances):
		r, p, y = transforms3d.euler.mat2euler(data[:, :, i], 'sxyz')
		gt_roll_pitch_yaw[0, i] = r
		gt_roll_pitch_yaw[1, i] = p
		gt_roll_pitch_yaw[2, i] = y


	numInstancesPred = predTimestamps.shape[1]
	pred_roll_pitch_yaw = np.zeros((3, numInstancesPred))
	for i in range(numInstancesPred):
		r, p, y = transforms3d.euler.quat2euler(predictions[i], 'sxyz')
		pred_roll_pitch_yaw[0, i] = r
		pred_roll_pitch_yaw[1, i] = p
		pred_roll_pitch_yaw[2, i] = y

	k = 0 # First index that matches - assumed that the others will match in sequence
	lastDiff = abs(viconTs[0] - predTimestamps[k])
	# Assuming there will be no index out of range
	while abs(viconTs[0] - predTimestamps[k + 1]) < lastDiff:
		lastDiff = abs(viconTs[0] - predTimestamps[k])
		k = k + 1

	lastIndex_gt = (numInstances) if (k + numInstances <= numInstancesPred) else (numInstancesPred-k)
	lastIndex_pred = (k + numInstances) if (k + numInstances <= numInstancesPred) else (numInstancesPred)
	numInstPlot = lastIndex_gt # number of instances being plotted

	plt.subplot(311)
	plt.plot(ts[0, 0:numInstPlot].reshape(numInstPlot, 1), gt_roll_pitch_yaw[0, 0:lastIndex_gt].reshape(numInstPlot, 1), 'k-')
	plt.plot(predTimestamps[0, k:lastIndex_pred].reshape(numInstPlot, 1), pred_roll_pitch_yaw[0, k:lastIndex_pred].reshape(numInstPlot, 1), 'r-')

	plt.subplot(312)
	plt.plot(ts[0, 0:numInstPlot].reshape(numInstPlot, 1), gt_roll_pitch_yaw[1, :].reshape(numInstPlot, 1), 'k-')
	plt.plot(predTimestamps[0, k:lastIndex_pred].reshape(numInstPlot, 1), pred_roll_pitch_yaw[1, k:lastIndex_pred].reshape(numInstPlot, 1), 'g-')

	plt.subplot(313)
	plt.plot(ts[0, 0:numInstPlot].reshape(numInstPlot, 1), gt_roll_pitch_yaw[2, :].reshape(numInstPlot, 1), 'k-')
	plt.plot(predTimestamps[0, k:lastIndex_pred].reshape(numInstPlot, 1), pred_roll_pitch_yaw[2, k:lastIndex_pred].reshape(numInstPlot, 1), 'r-')

	plt.show()
