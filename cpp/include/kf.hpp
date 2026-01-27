#ifndef KF_HPP
#define KF_HPP

#include <Eigen/Dense>
#include <iostream>

class KF {
public:
    /**
     * @brief Initialize the Kalman Filter
     * @param dim_x Dimension of the state vector
     * @param dim_z Dimension of the measurement vector
     */
    KF(int dim_x, int dim_z);

    /**
     * @brief Predict step
     * @param u Control input vector (optional, pass empty VectorXd if none)
     * @param F State transition matrix (optional, pass empty MatrixXd to use internal)
     * @param Q Process noise covariance (optional, pass empty MatrixXd to use internal)
     * @param B Control input matrix (optional, pass empty MatrixXd to use internal)
     */
    void predict(const Eigen::VectorXd& u = Eigen::VectorXd(), 
                 const Eigen::MatrixXd& F = Eigen::MatrixXd(), 
                 const Eigen::MatrixXd& Q = Eigen::MatrixXd(), 
                 const Eigen::MatrixXd& B = Eigen::MatrixXd());

    /**
     * @brief Update step
     * @param z Measurement vector
     * @param H Measurement matrix (optional, pass empty MatrixXd to use internal)
     * @param R Measurement noise covariance (optional, pass empty MatrixXd to use internal)
     */
    void update(const Eigen::VectorXd& z, 
                const Eigen::MatrixXd& H = Eigen::MatrixXd(), 
                const Eigen::MatrixXd& R = Eigen::MatrixXd());

    // Public state for direct access (matching Python style)
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd F;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd B;

    int dim_x;
    int dim_z;
    bool has_B; // Flag to track if B is set
};

#endif // KF_HPP
