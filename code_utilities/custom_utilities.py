from datetime import datetime

class Timer:
    def __init__(self):
        pass

    def getTimeString(self, mydate):
        return str(mydate.strftime('%Y-%m-%d %H:%M:%S'))

    def getTime(self):
        return datetime.now()

class MetaData:
    def __init__(self):
        self.activityCol = 1
        self.activities = [1,2,3,4,5,6,7,12,13,16,17]

        # 1 – lying (# subj - 8)
        # 2 – sitting (# subj - 8)
        # 3 – standing (# subj - 8)
        # 4 – walking (# subj - 8)
        # 5 – running (# subj - 6)
        # 6 – cycling (# subj - 7)
        # 7 – Nordic walking (# subj - 7)
        # 9 – watching TV (# subj - 1)
        # 10 – computer work  (# subj - 4)
        # 11 – car driving (# subj - 1)
        # 12 – ascending stairs  (# subj - 8)
        # 13 – descending stairs  (# subj - 8)
        # 16 – vacuum cleaning  (# subj - 8)
        # 17 – ironing (# subj - 8)
        # 18 – folding laundry  (# subj - 4)
        # 19 – house cleaning  (# subj - 5)
        # 20 – playing soccer (# subj - 2)
        # 24 – rope jumping (# subj - 6)

    def getRawColumnNames(self):
        columnNames = ['timestamp', 'activity_id', 'heartrate_bpm']

        imuColumns = ['%s_temp_c', '%s_3d_accel_16_1', '%s_3d_accel_16_2', '%s_3d_accel_16_3',
                      '%s_3d_accel_6_1', '%s_3d_accel_6_2', '%s_3d_accel_6_3',
                      '%s_3d_gyroscope_1', '%s_3d_gyroscope_2', '%s_3d_gyroscope_3',
                      '%s_3d_magnetometer_1', '%s_3d_magnetometer_2', '%s_3d_magnetometer_3',
                      '%s_orientation_1', '%s_orientation_2', '%s_orientation_3', '%s_orientation_4']

        # Prefixes to be appended to imuColumns
        prefix = ['hand', 'chest', 'ankle']

        # Append prefixes to columns
        for l in range(0, 3):
            for imu in imuColumns:
                imu = str(imu) % prefix[l]
                columnNames.append(imu)

        columnNames.append('subject')

        return columnNames

    def getOriginalColsDataType(self):
        dataType = [('timestamp', 'float'), ('activity_id', 'float'), ('heartrate_bpm', 'float'),
                    # imu hand - accel
                    ('hand_temp_c', 'float'),
                    ('hand_3d_accel_16g_1', 'float'), ('hand_3d_accel_16g_2', 'float'),
                    ('hand_3d_accel_16g_3', 'float'),
                    ('hand_3d_accel_6g_1', 'float'), ('hand_3d_accel_6g_2', 'float'), ('hand_3d_accel_6g_3', 'float'),
                    # imu hand - gyro, magneto
                    ('hand_3d_gyroscope_1', 'float'), ('hand_3d_gyroscope_2', 'float'),
                    ('hand_3d_gyroscope_3', 'float'),
                    ('hand_3d_magnetometer_1', 'float'), ('hand_3d_magnetometer_2', 'float'),
                    ('hand_3d_magnetometer_3', 'float'),
                    # imu hand - orientation
                    ('hand_orientation_1', 'float'), ('hand_orientation_2', 'float'), ('hand_orientation_3', 'float'),
                    ('hand_orientation_4', 'float'),
                    # imu chest - accel
                    ('chest_temp_c', 'float'),
                    ('chest_3d_accel_16g_1', 'float'), ('chest_3d_accel_16g_2', 'float'),
                    ('chest_3d_accel_16g_3', 'float'),
                    ('chest_3d_accel_6g_1', 'float'), ('chest_3d_accel_6g_2', 'float'),
                    ('chest_3d_accel_6g_3', 'float'),
                    # imu chest - gyro, magneto
                    ('chest_3d_gyroscope_1', 'float'), ('chest_3d_gyroscope_2', 'float'),
                    ('chest_3d_gyroscope_3', 'float'),
                    ('chest_3d_magnetometer_1', 'float'), ('chest_3d_magnetometer_2', 'float'),
                    ('chest_3d_magnetometer_3', 'float'),
                    # imu chest - orientation
                    ('chest_orientation_1', 'float'), ('chest_orientation_2', 'float'),
                    ('chest_orientation_3', 'float'),
                    ('chest_orientation_4', 'float'),
                    # imu ankle - accel
                    ('ankle_temp_c', 'float'),
                    ('ankle_3d_accel_16g_1', 'float'), ('ankle_3d_accel_16g_2', 'float'),
                    ('ankle_3d_accel_16g_3', 'float'),
                    ('ankle_3d_accel_6g_1', 'float'), ('ankle_3d_accel_6g_2', 'float'),
                    ('ankle_3d_accel_6g_3', 'float'),
                    # imu ankle - gyro, magneto
                    ('ankle_3d_gyroscope_1', 'float'), ('ankle_3d_gyroscope_2', 'float'),
                    ('ankle_3d_gyroscope_3', 'float'),
                    ('ankle_3d_magnetometer_1', 'float'), ('ankle_3d_magnetometer_2', 'float'),
                    ('ankle_3d_magnetometer_3', 'float'),
                    # imu ankle - orientation
                    ('ankle_orientation_1', 'float'), ('ankle_orientation_2', 'float'),
                    ('ankle_orientation_3', 'float'),
                    ('ankle_orientation_4', 'float'),
                    ('subject', 'int')
                    ]
        return dataType
    def getProcessedColsDataType(self):
        dataType = [('timestamp', 'float'), ('activity_id', 'float'), ('heartrate_bpm', 'float'),
                    # imu hand - accel
                    ('hand_temp_c', 'float'),
                    ('hand_3d_accel_16g_1', 'float'), ('hand_3d_accel_16g_2', 'float'),
                    ('hand_3d_accel_16g_3', 'float'),
                    ('hand_3d_accel_6g_1', 'float'), ('hand_3d_accel_6g_2', 'float'), ('hand_3d_accel_6g_3', 'float'),
                    # imu hand - gyro, magneto
                    ('hand_3d_gyroscope_1', 'float'), ('hand_3d_gyroscope_2', 'float'),
                    ('hand_3d_gyroscope_3', 'float'),
                    ('hand_3d_magnetometer_1', 'float'), ('hand_3d_magnetometer_2', 'float'),
                    # imu chest - accel
                    ('chest_temp_c', 'float'),
                    ('chest_3d_accel_16g_1', 'float'), ('chest_3d_accel_16g_2', 'float'),
                    ('chest_3d_accel_16g_3', 'float'),
                    ('chest_3d_accel_6g_1', 'float'), ('chest_3d_accel_6g_2', 'float'),
                    ('chest_3d_accel_6g_3', 'float'),
                    # imu chest - gyro, magneto
                    ('chest_3d_gyroscope_1', 'float'), ('chest_3d_gyroscope_2', 'float'),
                    ('chest_3d_gyroscope_3', 'float'),
                    ('chest_3d_magnetometer_1', 'float'), ('chest_3d_magnetometer_2', 'float'),
                    ('chest_3d_magnetometer_3', 'float'),
                    # imu ankle - accel
                    ('ankle_temp_c', 'float'),
                    ('ankle_3d_accel_16g_1', 'float'), ('ankle_3d_accel_16g_2', 'float'),
                    ('ankle_3d_accel_16g_3', 'float'),
                    ('ankle_3d_accel_6g_1', 'float'), ('ankle_3d_accel_6g_2', 'float'),
                    ('ankle_3d_accel_6g_3', 'float'),
                    # imu ankle - gyro, magneto
                    ('ankle_3d_gyroscope_1', 'float'), ('ankle_3d_gyroscope_2', 'float'),
                    ('ankle_3d_gyroscope_3', 'float'),
                    ('ankle_3d_magnetometer_1', 'float'), ('ankle_3d_magnetometer_2', 'float'),
                    ('ankle_3d_magnetometer_3', 'float'),
                    ('subject', 'int')
                    ]
        return dataType