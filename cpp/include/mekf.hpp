#ifndef MEKF_HPP
#define MEKF_HPP

#include <Eigen/Dense>
#include <iostream>
#include <functional>


class MEKF {
public:
    /**
     * @brief Initialize the Multiplicative Extended Kalman Filter
     * @param dim_x Dimension of the state vector
     * @param dim_z Dimension of the measurement vector
     */
    MEKF(int dim_x, int dim_z);

    // Typedefs for the functions
    // f(x, u) -> x_new
    using StateTransitionFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& x, const Eigen::VectorXd& u)>;
    // Jacobian of f w.r.t x -> F matrix
    using StateJacobianFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& x, const Eigen::VectorXd& u)>;
    
    // h(x) -> z_pred
    using MeasurementFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd& x)>;
    // Jacobian of h w.r.t x -> H matrix
    using MeasurementJacobianFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd& x)>;

    /**
     * @brief Predict step
     * @param fx State transition function f(x, u)
     * @param F_jac Function returning Jacobian F(x, u)
     * @param u Control input vector (optional)
     * @param Q Process noise covariance (optional, uses internal if empty)
     */
    void predict(StateTransitionFunc fx, 
                 StateJacobianFunc F_jac,
                 const Eigen::VectorXd& u = Eigen::VectorXd(),
                 const Eigen::MatrixXd& Q_in = Eigen::MatrixXd());

    /**
     * @brief Update step
     * @param z Measurement vector
     * @param hx Measurement function h(x)
     * @param H_jac Function returning Jacobian H(x)
     * @param R Measurement noise covariance (optional, uses internal if empty)
     */
    void update(const Eigen::VectorXd& z, 
                MeasurementFunc hx,
                MeasurementJacobianFunc H_jac, 
                const Eigen::MatrixXd& R_in = Eigen::MatrixXd());

    // Public state
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    int dim_x;
    int dim_z;
};

#endif // EKF_HPP
