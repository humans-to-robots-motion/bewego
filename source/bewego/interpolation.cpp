#include <bewego/interpolation.h>

#include <Eigen/Core>
#include <Eigen/LU> // for matrix inverse

namespace bewego {
double CalculateLocallyWeightedRegression(const Eigen::VectorXd& x_query,
                                          const Eigen::MatrixXd& X,
                                          const Eigen::VectorXd& Y,
                                          const Eigen::MatrixXd& D,
                                          double ridge_lambda) {
  // Default value is 0. The calculation uses ridge regression with a finite
  // regularizer, so the values should diminish smoothly to 0 away from the
  // data set anyway.
  if (Y.size() == 0) return 0.;

  // The "augmented" version of X has an extra 
  // constant feature to represent the bias.
  Eigen::MatrixXd Xaug(X.rows(), X.cols() + 1);
  Xaug << X, Eigen::VectorXd::Ones(Xaug.rows());

  Eigen::VectorXd x_query_aug(x_query.size() + 1);
  x_query_aug << x_query, 1;

  // Compute weighted points: WX, where W is the diagonal matrix of weights.
  Eigen::MatrixXd WX(Xaug.rows(), Xaug.cols());
  for (int i = 0; i < X.rows(); i++) {
    Eigen::VectorXd diff = x_query - X.row(i).transpose();
    double w = exp(-.5 * diff.transpose() * D * diff);
    WX.row(i) = w * Xaug.row(i);
  }

  // Fit plane to the weighted data
  Eigen::MatrixXd diag =
      (ridge_lambda * Eigen::VectorXd::Ones(Xaug.cols())).asDiagonal();

  // Calculate Pinv = X'WX + lambda I. P = inv(Pinv) is then
  // P = inv(X'WX + lambda I).
  Eigen::MatrixXd Pinv = WX.transpose() * Xaug + diag;
  Eigen::MatrixXd P = Pinv.inverse();
  Eigen::VectorXd beta =
      P * WX.transpose() * Y;  // beta = inv(X'WX + lambda I)WX'Y

  // Return inner product between plane and querrie point
  return beta.transpose() * x_query_aug;
}

std::vector<Eigen::VectorXd> LWR::ForwardMultiQuerry(
    const std::vector<Eigen::VectorXd>& xs) const {
  std::vector<Eigen::VectorXd> ys(xs.size());
  for (uint32_t i = 0; i < ys.size(); i++) {
    ys[i] = Forward(xs[i]);
  }
  return ys;
}

std::vector<Eigen::MatrixXd> LWR::JacobianMultiQuerry(
    const std::vector<Eigen::VectorXd>& xs) const {
  std::vector<Eigen::MatrixXd> Js(xs.size());
  for (uint32_t i = 0; i < Js.size(); i++) {
    Js[i] = Jacobian(xs[i]);
  }
  return Js;
}

}  // namespace bewego