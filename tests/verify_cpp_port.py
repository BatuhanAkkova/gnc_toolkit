import numpy as np

class KF:
    """
    Standard Linear Kalman Filter (KF).
    Suitable for linear estimation and navigation (e.g., constant velocity models).
    """
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.B = None

    def predict(self, u=None, F=None, Q=None, B=None):
        if F is None: F = self.F
        if Q is None: Q = self.Q
        if B is None: B = self.B

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        # P = FPF' + Q
        self.P = np.dot(np.dot(F, self.P), F.T) + Q

    def update(self, z, H=None, R=None):
        if H is None: H = self.H
        if R is None: R = self.R

        # y = z - Hx
        y = z - np.dot(H, self.x)

        # S = HPH' + R
        S = np.dot(np.dot(H, self.P), H.T) + R

        # K = PH'S^-1
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # x = x + Ky
        self.x = self.x + np.dot(K, y)

        # P = (I - KH)P(I - KH)' + KRK' (Joseph Form for stability)
        I = np.eye(self.dim_x)
        I_KH = I - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, R), K.T)

def run_test():
    dim_x = 2
    dim_z = 1
    
    kf = KF(dim_x, dim_z)
    
    # Initialize state
    kf.x = np.array([0., 0.])
    
    # F: Constant velocity
    kf.F = np.array([[1., 1.],
                     [0., 1.]])
                     
    # H: Measure position only
    kf.H = np.array([[1., 0.]])
    
    # Q: Process noise
    kf.Q = np.array([[0.1, 0.],
                     [0., 0.1]])
                     
    # R: Measurement noise
    kf.R = np.eye(1) * 1.0
    
    # P: Initial uncertainty
    kf.P = np.array([[10., 0.],
                     [0., 10.]])

    print(f"Initial State: \n{kf.x}")

    measurements = [1.1, 2.1, 3.2]

    for z_val in measurements:
        kf.predict()
        
        z = np.array([z_val])
        kf.update(z)
        
        print(f"Measurement: {z_val}")
        print(f"State: {kf.x}")
        # print(f"Covariance: \n{kf.P}")
        print("-----------------")

if __name__ == "__main__":
    run_test()
