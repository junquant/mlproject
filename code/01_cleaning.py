import numpy as np

# file properties
# -----------------------------------------------------
filePath = '../data/consolidated.txt'

dataType = [('timestamp','float'),('activity_id','int'),('heartrate_bpm','float'),
            # imu hand - accel
            ('hand_temp_c','float'),
            ('hand_3d_accel_16g_1','float'),('hand_3d_accel_16g_2', 'float'),('hand_3d_accel_16g_3','float'),
            ('hand_3d_accel_6g_1','float'),('hand_3d_accel_6g_2', 'float'),('hand_3d_accel_6g_3','float'),
            # imu hand - gyro, magneto
            ('hand_3d_gyroscope_1','float'),('hand_3d_gyroscope_2','float'),('hand_3d_gyroscope_3','float'),
            ('hand_3d_magnetometer_1', 'float'),('hand_3d_magnetometer_2','float'),('hand_3d_magnetometer_3','float'),
            # imu hand - orientation
            ('hand_orientation_1','float'),('hand_orientation_2','float'), ('hand_orientation_3', 'float'),('hand_orientation_4','float'),
            # imu chest - accel
            ('chest_temp_c', 'float'),
            ('chest_3d_accel_16g_1', 'float'), ('chest_3d_accel_16g_2', 'float'), ('chest_3d_accel_16g_3', 'float'),
            ('chest_3d_accel_6g_1', 'float'), ('chest_3d_accel_6g_2', 'float'), ('chest_3d_accel_6g_3', 'float'),
            # imu chest - gyro, magneto
            ('chest_3d_gyroscope_1', 'float'), ('chest_3d_gyroscope_2', 'float'), ('chest_3d_gyroscope_3', 'float'),
            ('chest_3d_magnetometer_1', 'float'), ('chest_3d_magnetometer_2', 'float'),
            ('chest_3d_magnetometer_3', 'float'),
            # imu chest - orientation
            ('chest_orientation_1', 'float'), ('chest_orientation_2', 'float'), ('chest_orientation_3', 'float'),
            ('chest_orientation_4', 'float'),
            # imu ankle - accel
            ('ankle_temp_c', 'float'),
            ('ankle_3d_accel_16g_1', 'float'), ('ankle_3d_accel_16g_2', 'float'), ('ankle_3d_accel_16g_3', 'float'),
            ('ankle_3d_accel_6g_1', 'float'), ('ankle_3d_accel_6g_2', 'float'), ('ankle_3d_accel_6g_3', 'float'),
            # imu ankle - gyro, magneto
            ('ankle_3d_gyroscope_1', 'float'), ('ankle_3d_gyroscope_2', 'float'), ('ankle_3d_gyroscope_3', 'float'),
            ('ankle_3d_magnetometer_1', 'float'), ('ankle_3d_magnetometer_2', 'float'),
            ('ankle_3d_magnetometer_3', 'float'),
            # imu ankle - orientation
            ('ankle_orientation_1', 'float'), ('ankle_orientation_2', 'float'), ('ankle_orientation_3', 'float'),
            ('ankle_orientation_4', 'float'),
            ('subject', 'float')
            ]

activityData = np.genfromtxt(filePath, delimiter = ',', skip_header = 1, dtype=dataType)

print(activityData.shape)
# Perform a summary of the data

# Visualization

# Impute using mean

# PCA
