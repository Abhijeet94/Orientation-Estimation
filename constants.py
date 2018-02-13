import os

DATA_FOLDER = 'ESE 650 Project 2'
CAM_FOLDER = os.path.join(DATA_FOLDER,'cam')
IMU_FOLDER = os.path.join(DATA_FOLDER,'imu')
VICON_FOLDER = os.path.join(DATA_FOLDER,'vicon')


# decent parameters

# positionCovParam = 0.05, 0.000100
# angularVelocityCovParam = 0.05, 0.05
# Q_processNoiseCovariance = np.diag(np.concatenate((positionCovParam * np.ones(3), angularVelocityCovParam * np.ones(3))))

# # 6 X 6
# accCovParam = 0.7, 0.1
# gyroCovParam = 0.05, 0.05
# R_measurementNoiseCov = np.diag(np.concatenate((accCovParam * np.ones(3), gyroCovParam * np.ones(3))))

# # 6 X 6
# orientationCovParam = 0.1, 0.0001
# angVelCovParam = 0.01 0.01
# P_prevCovariance_P_km1 = np.diag(np.concatenate((orientationCovParam * np.ones(3), angVelCovParam * np.ones(3))))