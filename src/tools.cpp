#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  if (estimations.empty() || estimations.size() != ground_truth.size()) {
     throw std::invalid_argument("Invalid dimensions of inputs");
  }

  VectorXd sumSq = VectorXd::Zero(estimations[0].size());
  for(int i = 0; i < estimations.size(); i++) {
    VectorXd diff = estimations[i] - ground_truth[i];
    sumSq = sumSq.array() + diff.array() * diff.array();
  }
  return (sumSq / estimations.size()).array().sqrt();
}