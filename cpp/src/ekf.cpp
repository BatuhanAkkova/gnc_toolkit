#include "ekf.hpp"

EKF::EKF(int dim_x, int dim_z) : dim_x(dim_x), dim_z(dim_z) {
    x = Eigen::VectorXd::Zero(dim_x);
    P = Eigen::MatrixXd::Identity(dim_x, dim_x);
    Q = Eigen::MatrixXd::Identity(dim_x, dim_x);
    R = Eigen::MatrixXd::Identity(dim_z, dim_z);
}

void EKF::predict(StateTransitionFunc fx, StateJacobianFunc F_jac, const Eigen::VectorXd& u, const Eigen::MatrixXd& Q_in) {
    // 1. Predict State: x = f(x, u)
    x = fx(x, u);

    // 2. Get Jacobian: F = F_jac(x, u)
    Eigen::MatrixXd F = F_jac(x, u);

    // 3. Predict Covariance: P = FPF' + Q
    const Eigen::MatrixXd& Q_curr = (Q_in.size() > 0) ? Q_in : Q;
    P = F * P * F.transpose() + Q_curr;
}

void EKF::update(const Eigen::VectorXd& z, MeasurementFunc hx, MeasurementJacobianFunc H_jac, const Eigen::MatrixXd& R_in) {
    // 1. Get Measurement Prediction: z_pred = h(x)
    Eigen::VectorXd z_pred = hx(x);

    // 2. Innovation: y = z - z_pred
    Eigen::VectorXd y = z - z_pred;

    // 3. Get Jacobian: H = H_jac(x)
    Eigen::MatrixXd H = H_jac(x);

    // 4. Innovation Covariance: S = HPH' + R
    const Eigen::MatrixXd& R_curr = (R_in.size() > 0) ? R_in : R;
    Eigen::MatrixXd S = H * P * H.transpose() + R_curr;

    // 5. Kalman Gain: K = PH'S^-1
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();

    // 6. State Correction: x = x + Ky
    x = x + K * y;

    // 7. Covariance Correction: P = (I - KH)P(I - KH)' + KRK' (Joseph Form)
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_x, dim_x);
    Eigen::MatrixXd I_KH = I - K * H;
    P = I_KH * P * I_KH.transpose() + K * R_curr * K.transpose();
}