#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param measurement_pack The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage measurement_pack);

  VectorXd GetCurrentState() const {
    return x_;
  }

private:
  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long previous_timestamp_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Number of sigma points
  int n_sig_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* Radar measurement noise covariance matrix
  MatrixXd R_laser_;

  ///* laser measurement noise covariance matrix
  MatrixXd R_radar_;

  ///* Laser measurement function
  MatrixXd H_;

  ///* counters for computing NIS
  int n_lidar_;
  int n_lidar_lt_7_815_;
  int n_radar_;
  int n_radar_lt_7_815_;

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Predict(const double &delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param measurement_pack The measurement at k+1
   */
  void UpdateLidar(const VectorXd &z);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param measurement_pack The measurement at k+1
   */
  void UpdateRadar(const VectorXd &z);

  /**
   * Predict next state and the next state covariance
   * matrix of a sigma point
   * @param delta_t Time between k and k+1 in s
   */
  VectorXd PredictSigmaAug(const VectorXd &sig_aug, const double &delta_t);

  /**
   * Normalize the radiant difference to [-pi, pi]
   * @param diff Difference between two states
   */
  VectorXd NormalizeStateDiff(const VectorXd &diff, const int &i);
};

#endif /* UKF_H */
