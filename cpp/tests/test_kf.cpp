#include "kf.hpp"
#include <iostream>
#include <vector>

int main() {
    // 1D Constant Velocity Model
    // State: [x, v]
    int dim_x = 2;
    int dim_z = 1;

    KF kf(dim_x, dim_z);

    // Initialize state
    kf.x << 0, 0; // Position 0, Velocity 0
    
    // F: Constant velocity
    // x_k+1 = x_k + v_k * dt
    // v_k+1 = v_k
    kf.F << 1, 1,
            0, 1;
            
    // H: Measure position only
    kf.H << 1, 0;
    
    // Q: Process noise
    kf.Q << 0.1, 0,
            0, 0.1;
            
    // R: Measurement noise
    kf.R << 1;
    
    // P: Initial uncertainty
    kf.P << 10, 0,
            0, 10;

    std::cout << "Initial State: \n" << kf.x.transpose() << std::endl;

    // Measurements: Object moving at v=1 starting at x=0.
    // True State:    0, 1, 2, 3
    // Measurements:  1.1, 2.1, 3.2 (noisy)
    std::vector<double> measurements = {1.1, 2.1, 3.2};

    for (double z_val : measurements) {
        kf.predict();
        
        Eigen::VectorXd z(1);
        z << z_val;
        kf.update(z);

        std::cout << "Measurement: " << z_val << "\n";
        std::cout << "State: " << kf.x.transpose() << "\n";
        // std::cout << "Covariance: \n" << kf.P << "\n";
        std::cout << "-----------------\n";
    }

    return 0;
}
