#include "kf.hpp"
#include <iostream>

KF::KF(int dim_x, int dim_z) : dim_x(dim_x), dim_z(dim_z), has_B(false) {
    x = Eigen::VectorXd::Zero(dim_x);
    P = Eigen::MatrixXd::Identity(dim_x, dim_x);
    F = Eigen::MatrixXd::Identity(dim_x, dim_x);
    H = Eigen::MatrixXd::Zero(dim_z, dim_x);
    Q = Eigen::MatrixXd::Identity(dim_x, dim_x);
    R = Eigen::MatrixXd::Identity(dim_z, dim_z);
    // B is uninitialized but can be resized later
    B = Eigen::MatrixXd(0, 0); 
}

void KF::predict(const Eigen::VectorXd& u, const Eigen::MatrixXd& F_in, const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& B_in) {
    // Use provided matrices or fallback to internal members
    const Eigen::MatrixXd& F_curr = (F_in.size() > 0) ? F_in : F;
    const Eigen::MatrixXd& Q_curr = (Q_in.size() > 0) ? Q_in : Q;
    const Eigen::MatrixXd& B_curr = (B_in.size() > 0) ? B_in : B;

    // x = Fx + Bu
    if (B_curr.size() > 0 && u.size() > 0) {
        x = F_curr * x + B_curr * u;
    } else {
        x = F_curr * x;
    }

    // P = FPF' + Q
    P = F_curr * P * F_curr.transpose() + Q_curr;
}

void KF::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H_in, const Eigen::MatrixXd& R_in) {
    const Eigen::MatrixXd& H_curr = (H_in.size() > 0) ? H_in : H;
    const Eigen::MatrixXd& R_curr = (R_in.size() > 0) ? R_in : R;

    // y = z - Hx
    Eigen::VectorXd y = z - H_curr * x;

    // S = HPH' + R
    Eigen::MatrixXd S = H_curr * P * H_curr.transpose() + R_curr;

    // K = PH'S^-1
    // Use LDLT or generic inverse. S is symmetric positive definite usually.
    Eigen::MatrixXd K = (P * H_curr.transpose()) * S.inverse();

    // x = x + Ky
    x = x + K * y;

    // P = (I - KH)P(I - KH)' + KRK' (Joseph Form for stability)
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_x, dim_x);
    Eigen::MatrixXd I_KH = I - K * H_curr;

    P = I_KH * P * I_KH.transpose() + K * R_curr * K.transpose();
}
