import numpy as np

class Drone(object):

    def __init__(self, filter_outliers=True, verbose=False):

        # H is the measurement model which casts next state onto an ovservation vector
        self.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0., 0., 0., 0.]], dtype=np.float64)

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
        Q[4:,4:] = np.diag([0.4, 0.2, 0.2, 0.5])
        return Q

    @staticmethod
    def observation_noise_covariance():
        """ 
        Output:
        R: 4x4 numpy array for the covariance matrix.
        """
        sigma = np.diag([0.1, 0.1, 0.1, 0.5])
        return sigma

    @staticmethod
    def observation(state):
        """ Implement the function h, from state to noise-less observation. 
        Input:
        state: (8,) numpy array representing state.
        Output:
        obs: (4,) numpy array representing observation.
        """
        return state[:4]

    def simulation(self, init_pose=[0.0, 0.0, 0.0, 0.0], T=100):
        """ simulate with fixed start state for T timesteps.
        Input:
        T: an integer (=100).
        Output:
        states: (T,8) numpy array of states, including the given start state.
        observations: (T,4) numpy array of observations, Including the observation of start state.
        """
        x_0 = np.array([init_pose[0], init_pose[1], init_pose[2], init_pose[3], 0.0, 0.0, 0.0, 0.0])
        states = [x_0]
        A = self.system_matrix()
        Q = self.process_noise_covariance()
        R = self.observation_noise_covariance()
        z_0 = self.observation(x_0) + np.random.multivariate_normal(np.zeros((R.shape[0],)), R)
        observations = [z_0]
        for t in range(1,T):
            process_noise = np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q)
            xt = np.dot(A, x_0) + process_noise
            zt = self.observation(xt) + np.random.multivariate_normal(np.zeros((R.shape[0],)), R)
            states.append(xt)
            observations.append(zt)
            x_0 = xt
            
        return np.array(states), np.array(observations)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(33)
    drone = Drone()
    states, observations = drone.simulation()

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Simulated Drone Trajectory')
    ax.scatter(states[:,0], states[:,1], states[:,2], c=np.arange(states.shape[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Observed Drone Trajectory')
    ax.scatter(observations[:,0], observations[:,1], observations[:,2], c=np.arange(states.shape[0]))
    plt.show()