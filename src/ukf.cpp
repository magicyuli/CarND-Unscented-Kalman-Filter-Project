#include "ukf.h"

#include <cmath>
#include <iostream>

#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_* std_laspy_;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_* std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;
  
  n_x_ = 5;
  n_aug_ = 7;
  n_sig_ = 2 * n_aug_ + 1;
  lambda_ = 3 - n_aug_;

  // init weights
  weights_ = VectorXd(n_sig_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_; i++) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  // init prediction of sigma points
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

  n_lidar_ = 0;
  n_lidar_lt_7_815_ = 0;
  n_radar_ = 0;
  n_radar_lt_7_815_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} measurement_pack The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    previous_timestamp_ = measurement_pack.timestamp_;

    // first measurement
    std::cout << "Initializing ... " << std::endl;

    // init x
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      x_ << measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]),
            measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]), 0, 0, 0;
    }
    else {
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    std::cout << "Initialized ... " << std::endl;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  Predict((measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(measurement_pack.raw_measurements_);
  }
  else {
    UpdateLidar(measurement_pack.raw_measurements_);
  }

  previous_timestamp_ = measurement_pack.timestamp_;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Predict(const double &delta_t) {
  // generate sigma points with augmentation,
  VectorXd x_aug(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd spread_A = sqrt(lambda_ + n_aug_) * A;

  // and predict the next state for sigma points
  Xsig_pred_.col(0) = PredictSigmaAug(x_aug, delta_t);
  for (int i = 0; i < n_aug_; i++) {
    Xsig_pred_.col(i + 1) = PredictSigmaAug(x_aug + spread_A.col(i), delta_t);
    Xsig_pred_.col(i + 1 + n_aug_) = PredictSigmaAug(x_aug - spread_A.col(i), delta_t);
  }

  // compute predicted mean and covariance
  x_ = Xsig_pred_ * weights_;

  MatrixXd tmp(n_x_, n_sig_);
  for (int i = 0; i < n_sig_; i++) {
    tmp.col(i) = NormalizeStateDiff(Xsig_pred_.col(i) - x_, 3);
  }
  MatrixXd tmp_t = tmp.transpose();
  for (int i = 0; i < n_sig_; i++) {
    tmp.col(i) = weights_(i) * tmp.col(i);
  }
  P_ = tmp * tmp_t;
}

VectorXd UKF::PredictSigmaAug(const VectorXd &sig_aug, const double &delta_t) {
  VectorXd sig_pred = sig_aug.head(5);
  double a_v = sig_aug(5);
  double a_yaw = sig_aug(6);
  double v = sig_pred(2);
  double yaw = sig_pred(3);
  double v_yaw = sig_pred(4);

  if (v_yaw == 0) {
    sig_pred(0) += v * cos(yaw) * delta_t;
    sig_pred(1) += v * sin(yaw) * delta_t;
  }
  else {
    sig_pred(0) += v * (sin(yaw + v_yaw * delta_t) - sin(yaw)) / v_yaw;
    sig_pred(1) += v * (-cos(yaw + v_yaw * delta_t) + cos(yaw)) / v_yaw;
  }
  sig_pred(3) += v_yaw * delta_t;
  
  VectorXd noise(5);
  noise << 0.5 * delta_t * delta_t * cos(yaw) * a_v,
           0.5 * delta_t * delta_t * sin(yaw) * a_v,
           delta_t * a_v,
           0.5 * delta_t * delta_t * a_yaw,
           delta_t * a_yaw;
  return sig_pred + noise;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} measurement_pack
 */
void UKF::UpdateLidar(const VectorXd &z) {
  // convert prediction to measurement space
  VectorXd z_pred = H_ * x_;
  MatrixXd H_t = H_.transpose();
  MatrixXd S = H_ * P_ * H_t + R_laser_;
  MatrixXd K = P_ * H_t * S.inverse();

  VectorXd diff = z - z_pred;
  //new estimate
  x_ += K * diff;

  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;

  // compute NIS
  MatrixXd nis = diff.transpose() * S.inverse() * diff;
  n_lidar_++;
  if (nis(0, 0) < 7.815) {
    n_lidar_lt_7_815_++;
  }
  std::cout << "% of lidar NIS < 7.815: " << n_lidar_lt_7_815_ / (float) n_lidar_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} measurement_pack
 */
void UKF::UpdateRadar(const VectorXd &z) {
  // convert prediction to measurement space
  MatrixXd Zsig_pred(3, n_sig_);
  for (int i = 0; i < n_sig_; i++) {
    double px = Xsig_pred_.col(i)(0);
    double py = Xsig_pred_.col(i)(1);
    double v = Xsig_pred_.col(i)(2);
    double yaw = Xsig_pred_.col(i)(3);
    VectorXd z_sig_i(3);
    z_sig_i << sqrt(px * px + py * py),
               atan2(py, px),
               (px * v * cos(yaw) + py * v * sin(yaw)) / sqrt(px * px + py * py);
    Zsig_pred.col(i) = z_sig_i;
  }
  //calculate mean predicted measurement
  VectorXd z_pred = Zsig_pred * weights_;
  //calculate covariance matrix in the measurement space
  MatrixXd tmp_Z(3, n_sig_);
  for (int i = 0; i < n_sig_; i++) {
    tmp_Z.col(i) = NormalizeStateDiff(Zsig_pred.col(i) - z_pred, 1);
  }
  MatrixXd tmp_Z_t = tmp_Z.transpose();
  for (int i = 0; i < n_sig_; i++) {
    tmp_Z.col(i) *= weights_(i);
  }
  
  // calculate Kalman gain denomenator
  MatrixXd S = tmp_Z * tmp_Z_t + R_radar_;

  MatrixXd tmp_X(n_x_, n_sig_);
  for (int i = 0; i < n_sig_; i++) {
    tmp_X.col(i) = weights_(i) * NormalizeStateDiff(Xsig_pred_.col(i) - x_, 3);
  }
  // calculate Kalman gain numerator
  MatrixXd Tc = tmp_X * tmp_Z_t;
  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  VectorXd diff = NormalizeStateDiff(z - z_pred, 1);
  // update state mean and covariance matrix
  x_ += K * diff;

  P_ -= K * S * K.transpose();

  // compute NIS
  MatrixXd nis = diff.transpose() * S.inverse() * diff;
  n_radar_++;
  if (nis(0, 0) < 7.815) {
    n_radar_lt_7_815_++;
  }
  std::cout << "% of radar NIS < 7.815: " << n_radar_lt_7_815_ / (float) n_radar_ << std::endl;
}

VectorXd UKF::NormalizeStateDiff(const VectorXd &diff, const int &i) {
  VectorXd res = diff;
  while (res(i) > M_PI) {
    res(i) -= 2 * M_PI;
  }
  while (res(i) < -M_PI) {
    res(i) += 2 * M_PI;
  }
  return res;
}