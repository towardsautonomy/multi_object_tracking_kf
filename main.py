import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

from simulation import Drone
from kalmanFilter import KalmanFilter
from plot_helper import draw_3d

if __name__ == "__main__":
    T = 100
    # simulate drone trajectory
    np.random.seed(33)
    drone1 = Drone()
    states1, observations1 = drone1.simulation(init_pose=[0.5, 0.0, 5.0, 90.0], T=T)

    np.random.seed(99)
    drone2 = Drone()
    states2, observations2 = drone2.simulation(init_pose=[4.5, 2.0, 3.0, 60.0], T=T)

    # instantiate a kalman filter
    kf = KalmanFilter(verbose=True)

    true_poses1, true_poses2 = [], []
    observed_poses = []
    tracked_poses1, tracked_poses2 = [], []
    tracked_states1, tracked_states2 = [], []
    for i in range(T):
        # measurement
        observation1 = {'class':'Drone', 'x':observations1[i,0], 'y':observations1[i,1], 'z':observations1[i,2], 'yaw':observations1[i,3]}
        observation2 = {'class':'Drone', 'x':observations2[i,0], 'y':observations2[i,1], 'z':observations2[i,2], 'yaw':observations2[i,3]}
        tracked_poses = kf.process([observation1, observation2])

        # true poses
        true_poses1.append(states1[i][:4])
        true_poses2.append(states2[i][:4])
        
        # observations
        observed_poses.append([observations1[i,0], observations1[i,1], observations1[i,2], observations1[i,3]])
        observed_poses.append([observations2[i,0], observations2[i,1], observations2[i,2], observations2[i,3]])
        for pose_ in tracked_poses:
            if pose_['id'] == 0:
                # append to list for plotting
                tracked_poses1.append([pose_['x'], pose_['y'], pose_['z'], pose_['yaw']])
                tracked_states1.append([[pose_['x'], pose_['y'], pose_['z'], pose_['yaw']], pose_['covar']])

            elif pose_['id'] == 1:
                # append to list for plotting
                tracked_poses2.append([pose_['x'], pose_['y'], pose_['z'], pose_['yaw']])
                tracked_states2.append([[pose_['x'], pose_['y'], pose_['z'], pose_['yaw']], pose_['covar']])

            print('[id,class]: [{},{}]'.format(pose_['id'], pose_['class']))
            print('  - [x,y,z,yaw]: [{},{},{},{}]'.format(
                pose_['x'], pose_['y'], pose_['z'], pose_['yaw']))
            print('  - [vx,vy,vz,vyaw]: [{},{},{},{}]'.format(
                pose_['vx'], pose_['vy'], pose_['vz'], pose_['vyaw']))

        time.sleep(0.033)

    # tear down the kalman filter
    kf.teardown()

    # convert to numpy array
    true_poses1, true_poses2 = np.array(true_poses1), np.array(true_poses2)
    observed_poses = np.array(observed_poses)
    tracked_poses1, tracked_poses2 = np.array(tracked_poses1), np.array(tracked_poses2)
    
    # plotting
    fig = plt.figure(figsize=(24,10))
    ax = fig.add_subplot(131, projection='3d')
    ax.set_title('Simulated Trajectory', fontsize=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(true_poses1[:,0], true_poses1[:,1], true_poses1[:,2], c=np.tile([1.,0.,0.], (true_poses1.shape[0],1)))
    ax.scatter(true_poses2[:,0], true_poses2[:,1], true_poses2[:,2], c=np.tile([0.,0.,1.], (true_poses2.shape[0],1)))
    ax.legend(['Drone1', 'Drone2'], fontsize=16)
    ax.view_init(elev=25., azim=-80)

    ax = fig.add_subplot(132, projection='3d')
    ax.set_title('Measured Poses', fontsize=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(observed_poses[:,0], observed_poses[:,1], observed_poses[:,2], c=np.arange(observed_poses.shape[0]), s=16)
    ax.view_init(elev=25., azim=-80)

    ax = fig.add_subplot(133, projection='3d')
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title('Tracked Trajectory', fontsize=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # trajectory
    ax.scatter(tracked_poses1[:,0], tracked_poses1[:,1], tracked_poses1[:,2], c=np.tile([1.,0.,0.], (tracked_poses1.shape[0],1)))
    ax.scatter(tracked_poses2[:,0], tracked_poses2[:,1], tracked_poses2[:,2], c=np.tile([0.,0.,1.], (tracked_poses2.shape[0],1)))
    # covariance
    for mu, covar in tracked_states1:
        draw_3d(ax, covar[:3,:3], mu[:3])
    for mu, covar in tracked_states2:
        draw_3d(ax, covar[:3,:3], mu[:3])
    ax.legend(['Drone1', 'Drone2'], fontsize=16)
    ax.view_init(elev=25., azim=-80)

    plt.savefig('media/tracked_trajectory.png')
    plt.show()