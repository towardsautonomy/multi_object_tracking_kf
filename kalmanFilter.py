import numpy as np
import time
import threading
import sys

class KalmanFilter(object):

    def __init__(self, filter_outliers=True, verbose=False):
        self.tracked_objects = []
        self.next_id = -1

        self.Q = self.process_noise_covariance()
        self.R = self.observation_noise_covariance()

        # H is the measurement model which casts next state onto an ovservation vector
        self.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 0.]], dtype=np.float64)

        # other parameters
        self.filter_outliers = filter_outliers
        self.verbose = verbose
        self.processing = False

        # start the timer thread
        self.thread_ = threading.Thread(target=self.timer_thread)
        self.thread_is_alive = True
        
        # Start the thread
        self.exit_event = threading.Event()
        self.daemon = True
        self.thread_.start()

    def get_new_id(self, max_id=9999):
        self.next_id = ((self.next_id + 1) % (max_id + 1))
        return self.next_id

    def add_new_object(self, obj_class, x, y, z, yaw):
        if self.verbose:
            print('Adding new object: {}'.format(obj_class))

        state = [x, y, z, yaw, 0., 0., 0., 0.]
        state_sigma = np.zeros((8,8), dtype=np.float32)
        state_sigma[:4,:4] = self.R
        obj = {'id':self.get_new_id(), 'class':obj_class, 
                'state_mu':state, 'state_sigma':state_sigma, 'timestamp':time.time()}
        self.tracked_objects.append(obj)

    def find_matching_object_idx(self, obj_class, x, y, z, yaw):
        return_idx = -1
        min_dist = 1e5
        for i, tracked_obj in enumerate(self.tracked_objects):
            if obj_class == tracked_obj['class']:
                dist = np.sqrt((x - tracked_obj['state_mu'][0]) ** 2 + \
                               (y - tracked_obj['state_mu'][1]) ** 2 + \
                               (z - tracked_obj['state_mu'][2]) ** 2 )
                
                variance_xyz = [tracked_obj['state_sigma'][0,0], tracked_obj['state_sigma'][1,1], tracked_obj['state_sigma'][2,2]]
                sigma_dist = np.sqrt(variance_xyz[0] + variance_xyz[1] + variance_xyz[2])

                if dist < min_dist and dist < 3*sigma_dist:
                    min_dist = dist
                    # match found
                    return_idx = i

        return return_idx

    def timer_thread(self):
        # If the child thread is still running
        while self.thread_is_alive:
            # wait if processing new data
            while self.processing:
                time.sleep(0.001)
            
            # proceed
            duration = 0.1 # 100 milliseconds
            time.sleep(duration) # Sleep
            
            # iterate through each tracked object
            for i, tracked_obj in enumerate(self.tracked_objects):
                dt = time.time() - tracked_obj['timestamp']
                if dt > duration:
                    if self.verbose:
                        print('Removing tracked object [{}] with id: {}'.format(tracked_obj['class'], tracked_obj['id']))
                    del self.tracked_objects[i]

    def teardown(self):
        if self.verbose:
            print('Tearing down the Kalman Filter.')

        self.thread_is_alive = False

    @staticmethod
    def system_matrix(dt=0.033):
        """ 
        Output:
        A: 8x8 numpy array for the system matrix.
        """
        A = np.array([[1., 0., 0., 0., dt, 0., 0., 0.],
                      [0., 1., 0., 0., 0., dt, 0., 0.],
                      [0., 0., 1., 0., 0., 0., dt, 0.],
                      [0., 0., 0., 1., 0., 0., 0., dt],
                      [0., 0., 0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 1.]], dtype=np.float64)
        return A

    @staticmethod
    def process_noise_covariance():
        """ 
        Output:
        Q: 8x8 numpy array for the covariance matrix.
        """
        Q = np.zeros((8,8), dtype=np.float64)
        Q[4:,4:] = np.diag([0.2, 0.2, 0.2, 0.5])
        return Q

    @staticmethod
    def observation_noise_covariance():
        """ 
        Output:
        R: 4x4 numpy array for the covariance matrix.
        """
        sigma = np.diag([1.0, 1.0, 1.0, 2.0])
        return sigma

    def process(self, measurements):
        """
        Input:
        measurements: list of dictionary of measurement [class, x, y, z, yaw]
        """ 
        tracked_objects = []
        for measurement in measurements:   
            self.processing = True
            timestamp = time.time()
            tracked_obj = None
            # find association
            associated_idx = self.find_matching_object_idx(
                                    measurement['class'], measurement['x'], measurement['y'], 
                                    measurement['z'], measurement['yaw']
                                    )
            if associated_idx != -1:
                dt = timestamp - self.tracked_objects[associated_idx]['timestamp']
                # print(dt)
                A = self.system_matrix(dt)

                # Prediction
                mu_bar_next = np.dot(A, self.tracked_objects[associated_idx]['state_mu'])
                sigma_bar_next = np.dot(A, np.dot(self.tracked_objects[associated_idx]['state_sigma'], A.T)) \
                                        + self.Q

                # compute Kalman Gain
                kalman_gain_numerator = np.dot(sigma_bar_next, self.H.T)
                kalman_gain_denominator = np.dot(self.H, np.dot(sigma_bar_next, self.H.T)) + self.R # this is the innovation covariance matrix, S
                kalman_gain = np.dot(kalman_gain_numerator, np.linalg.inv(kalman_gain_denominator))

                # Correction
                observation = [measurement['x'], measurement['y'], measurement['z'], measurement['yaw']]
                expected_observation = np.dot(self.H, mu_bar_next)


                # let's compute Mahalanobis distance
                S = kalman_gain_denominator
                deviation = np.sqrt(np.dot((observation - expected_observation).T, np.dot(np.linalg.inv(S), (observation - expected_observation))))

                # outlier rejection
                if not self.filter_outliers or deviation <= 1.5:
                    mu_next = mu_bar_next + np.dot(kalman_gain, (observation - expected_observation).T)
                    sigma_next = np.dot((np.eye(8, dtype=np.float64) - np.dot(kalman_gain, self.H)), sigma_bar_next)
                    # update timestamp only if the sample is an inlier
                    self.tracked_objects[associated_idx]['timestamp'] = timestamp
                else:
                    mu_next = mu_bar_next
                    sigma_next = sigma_bar_next

                # Update State
                self.tracked_objects[associated_idx]['state_mu'] = mu_next
                self.tracked_objects[associated_idx]['state_sigma'] = sigma_next

                tracked_obj = {'id':self.tracked_objects[associated_idx]['id'],
                            'class':self.tracked_objects[associated_idx]['class'], 
                            'x':self.tracked_objects[associated_idx]['state_mu'][0], 
                            'y':self.tracked_objects[associated_idx]['state_mu'][1], 
                            'z':self.tracked_objects[associated_idx]['state_mu'][2], 
                            'yaw':self.tracked_objects[associated_idx]['state_mu'][3],
                            'vx':self.tracked_objects[associated_idx]['state_mu'][4], 
                            'vy':self.tracked_objects[associated_idx]['state_mu'][5], 
                            'vz':self.tracked_objects[associated_idx]['state_mu'][6], 
                            'vyaw':self.tracked_objects[associated_idx]['state_mu'][7],
                            'covar':self.tracked_objects[associated_idx]['state_sigma']}

            else:
                # new object
                self.add_new_object(
                                    measurement['class'], measurement['x'], measurement['y'], 
                                    measurement['z'], measurement['yaw']
                                    )

                tracked_obj = {'id':self.tracked_objects[-1]['id'],
                            'class':self.tracked_objects[-1]['class'], 
                            'x':self.tracked_objects[-1]['state_mu'][0], 
                            'y':self.tracked_objects[-1]['state_mu'][1], 
                            'z':self.tracked_objects[-1]['state_mu'][2], 
                            'yaw':self.tracked_objects[-1]['state_mu'][3],
                            'vx':self.tracked_objects[associated_idx]['state_mu'][4], 
                            'vy':self.tracked_objects[associated_idx]['state_mu'][5], 
                            'vz':self.tracked_objects[associated_idx]['state_mu'][6], 
                            'vyaw':self.tracked_objects[associated_idx]['state_mu'][7],
                            'covar':self.tracked_objects[associated_idx]['state_sigma']}

            self.processing = False
            tracked_objects.append(tracked_obj)
        return tracked_objects

if __name__ == '__main__':
    kf = KalmanFilter(verbose=True)

    for i in range(100):
        observations = []
        # measurement
        observation = {'class':'Car', 'x':0.5+i*0.1, 'y':0.3, 'z':2.2, 'yaw':180.0}
        observations.append(observation)
        observation = {'class':'Ped', 'x':2.5+i*0.1, 'y':0.3, 'z':2.2, 'yaw':180.0}
        observations.append(observation)
        observations = kf.process(observations)

        for observation in observations:
            print('[id,class]: [{},{}]'.format(observation['id'], observation['class']))
            print('  - [x,y,z,yaw]: [{},{},{},{}]'.format(
                observation['x'], observation['y'], observation['z'], observation['yaw']))
            print('  - [vx,vy,vz,vyaw]: [{},{},{},{}]'.format(
                observation['vx'], observation['vy'], observation['vz'], observation['vyaw']))
        time.sleep(0.033)

    # measurement
    time.sleep(0.2)
    observation = {'class':'Car', 'x':0.5+1.0, 'y':0.3, 'z':2.2, 'yaw':180.0}
    observations = kf.process([observation])
    for observation in observations:
        print('[id,class]: [{},{}]'.format(observation['id'], observation['class']))
        print('  - [x,y,z,yaw]: [{},{},{},{}]'.format(
            observation['x'], observation['y'], observation['z'], observation['yaw']))
        print('  - [vx,vy,vz,vyaw]: [{},{},{},{}]'.format(
            observation['vx'], observation['vy'], observation['vz'], observation['vyaw']))

    # measurement
    time.sleep(0.2)
    observation = {'class':'Car', 'x':0.5+1.1, 'y':0.3, 'z':2.2, 'yaw':180.0}
    observations = kf.process([observation])
    for observation in observations:
        print('[id,class]: [{},{}]'.format(observation['id'], observation['class']))
        print('  - [x,y,z,yaw]: [{},{},{},{}]'.format(
            observation['x'], observation['y'], observation['z'], observation['yaw']))
        print('  - [vx,vy,vz,vyaw]: [{},{},{},{}]'.format(
            observation['vx'], observation['vy'], observation['vz'], observation['vyaw']))
    kf.teardown()