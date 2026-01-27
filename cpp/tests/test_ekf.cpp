#include "ekf.hpp"
#include <iostream>
#include <cmath>
#include <vector>

// Simple Radar Measurement Example
// State: [position, velocity] (1D)
// Measurement: Range (r) = sqrt(x^2 + alt^2) ... wait, 1D? 
// Let's do a simple non-linear measurement: z = x^2 / 20 (arbitrary non-linear)
// Or better: Standard Radar: Range = sqrt(pos^2 + const^2)

int main() {
    int dim_x = 2; // pos, vel
    int dim_z = 1; // range

    EKF ekf(dim_x, dim_z);

    // Initial State
    ekf.x << 0, 10; // Pos=0, Vel=10
    ekf.P << 10, 0,
             0, 10;
    
    // Process noise
    ekf.Q << 0.1, 0,
             0, 0.1;
    
    // Measurement noise
    ekf.R << 1.0;

    double dt = 0.1;
    
    // 1. Define State Transition f(x, u)
    // Linear Motion: x_new = x + v*dt, v_new = v
    auto fx = [dt](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        Eigen::VectorXd x_new(2);
        x_new(0) = x(0) + x(1) * dt;
        x_new(1) = x(1);
        return x_new;
    };

    // Jacobian F(x, u)
    auto F_jac = [dt](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::MatrixXd {
        Eigen::MatrixXd F(2, 2);
        F << 1, dt,
             0, 1;
        return F;
    };

    // 2. Define Measurement Function h(x)
    // Radar at altitude A=100 measuring slant range to object at 'pos'
    // range = sqrt(pos^2 + 100^2)
    double altitude = 100.0;
    auto hx = [altitude](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        double pos = x(0);
        Eigen::VectorXd z(1);
        z(0) = std::sqrt(pos*pos + altitude*altitude);
        return z;
    };

    // Jacobian H(x) = d(range)/d(pos), d(range)/d(vel)
    // d(sqrt(x^2+A^2))/dx = (1/2)*(x^2+A^2)^(-1/2) * 2x = x / sqrt(x^2+A^2)
    auto H_jac = [altitude](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        double pos = x(0);
        double range = std::sqrt(pos*pos + altitude*altitude);
        Eigen::MatrixXd H(1, 2);
        H(0, 0) = pos / range;
        H(0, 1) = 0; // Velocity doesn't affect range directly
        return H;
    };

    std::cout << "Initial State: " << ekf.x.transpose() << std::endl;

    // Simulate 10 steps
    for (int i = 0; i < 10; ++i) {
        // True Physics
        // Pos = 0 + 10*t
        double true_pos = 10.0 * (i+1) * dt;
        double true_range = std::sqrt(true_pos*true_pos + altitude*altitude);
        // Add noise? Let's keep it clean for simple check or add deterministic noise
        double zs = true_range; 

        // PREDICT
        ekf.predict(fx, F_jac);

        // UPDATE
        Eigen::VectorXd z(1);
        z << zs;
        ekf.update(z, hx, H_jac);

        std::cout << "Step " << i << ": TruePos=" << true_pos 
                  << " EstPos=" << ekf.x(0) 
                  << " EstVel=" << ekf.x(1) 
                  << " Meas=" << zs << std::endl;
    }

    return 0;
}
