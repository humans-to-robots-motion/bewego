// Copyright (c) 2016, Max Planck Institute.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/util/interpolation.h>
#include <bewego/util/misc.h>
#include <bewego/util/tricubic_interpolation.h>
#include <gtest/gtest.h>

#include <Eigen/LU>

using namespace bewego;
using std::cout;
using std::endl;

#define TEST_DELTA 0.3
#define TEST_MAX 1.0

void GetData(std::shared_ptr<const AffineMap> f, Eigen::MatrixXd *X,
             Eigen::VectorXd *Y) {
  double delta = TEST_DELTA;
  double max = TEST_MAX;
  uint32_t nb_points_per_dim = max / delta + 1;
  uint32_t nb_points =
      nb_points_per_dim * nb_points_per_dim * nb_points_per_dim;
  Eigen::MatrixXd data_i(nb_points, 3);
  Eigen::VectorXd data_o(nb_points);

  uint32_t i = 0;
  for (double x = 0.; x < max; x += delta) {
    for (double y = 0.; y < max; y += delta) {
      for (double z = 0.; z < max; z += delta) {
        Eigen::Vector3d p;
        p.x() = x;
        p.y() = y;
        p.z() = z;
        data_i.row(i) = p;
        data_o(i) = (*f)(p)[0];
        i++;
      }
    }
  }

  // cout << "nb_points : " << nb_points << " (i = " << i << ")" << endl;
  (*X) = data_i;
  (*Y) = data_o;
}

std::shared_ptr<LinearMap> SetUpLinearFunction() {
  Eigen::MatrixXd a(1, 3);
  a << 1, 2, 3;
  return std::make_shared<LinearMap>(a);
}

std::shared_ptr<ConstantMap> SetUpConstantFunction() {
  Eigen::VectorXd a(1);
  a << 17;
  return std::make_shared<ConstantMap>(1, a);
}

// Linear Functions Test
bool CheckLinearFunction() {
  std::shared_ptr<const AffineMap> f = SetUpLinearFunction();
  Eigen::MatrixXd X;
  Eigen::VectorXd Y;
  GetData(f, &X, &Y);

  // TODO don't know yet how to test the regularizers
  double lambda_1 = 0.;  // L2 norm regularizer scalar
  double lambda_2 = 0.;  // Proximal regularizer scalar
  auto regressor = std::make_shared<const LinearRegressor>(lambda_1, lambda_2);
  Eigen::VectorXd w = regressor->ComputeParameters(X, Y);

  // Print regressed coefficients
  // cout << "(function)   b : " << f->b().transpose() << endl;
  // cout << "(regressed)  w : " << w.transpose() << endl;

  double min = 0;
  double max = TEST_MAX;
  for (int i = 0; i < 1000; i++) {
    Eigen::Vector3d p = bewego::util::Random(3);
    p.x() = min + p.x() * max;
    p.y() = min + p.y() * max;
    p.z() = min + p.z() * max;

    double error = (w.transpose() * p) - (*f)(p)[0];
    if (std::fabs(error) > 1e-6) {
      return false;
    }
  }
  return true;
}

TEST(LinearRegression, CheckNbPoints) {
  double delta = TEST_DELTA;
  double max = TEST_MAX;
  uint32_t nb_points_per_dim = max / delta + 1;
  uint32_t i = 0;
  for (double x = 0.; x < max; x += delta) {
    i++;
  }
  ASSERT_EQ(nb_points_per_dim, i);
}

TEST(LinearRegression, Main) { ASSERT_TRUE(CheckLinearFunction()); }

// Matlab test:
//
// % Data
// X = [-1, .5, 0, .5, 1]'
// Y = [.1, .2, .3, .4, .5]'
// x_query = .3
// lambda = .1
//
// % Augment with constant feature
// Xa = [X, ones(size(X,1),1)]
//
// % Calculate linear function
// W = diag(exp(-.5*(x_query - X).^2))
// beta = inv(Xa'*W*Xa+lambda*eye(2,2))*Xa'*W*Y
//
// beta =
//    0.166173
//    0.257212
//
// % Evaluate qt query point
// beta'*[x_query; 1]
// ans = 0.307063690815270
TEST(LWRTest, Evaluation) {
  // Setup data.
  Eigen::MatrixXd X(5, 1);
  X << -1, .5, 0, .5, 1;
  Eigen::VectorXd Y(5);
  Y << .1, .2, .3, .4, .5;

  Eigen::VectorXd x_query(1);
  x_query << .3;
  double lambda = .1;

  Eigen::MatrixXd D = Eigen::MatrixXd::Identity(1, 1);
  double value = CalculateLocallyWeightedRegression(x_query, X, Y, D, lambda);
  // cout << value << endl;
  EXPECT_NEAR(value, 0.307063690815270, 1e-7);
}

/**
class TriCubicTest : public ::testing::Test {
 public:
  void SetUp() {
    analytical_grid_ = AnalyticalGrid(
        .02, CreateEnvironmentBox(-.1, .1, -.1, .1, -.1, .1), false);
    Eigen::VectorXd a(3);
    a << 1, 2, 3;
    double b = -.1;
    linear_function_ = SetUpLinearFunction();
    constant_function_ = SetUpConstantFunction();
  }

  void ValidateGridPoint(const Eigen::Vector3i &query_cell, double precision,
                         bool verbose) {
    Eigen::Vector3d query_pt = analytical_grid_.gridToWorld(query_cell);
    double actual_value = (*linear_function_)(query_pt)[0];
    double potential_value = analytical_grid_.CalculatePotential(query_pt);
    if (verbose) {
      cout << "------------------------" << endl;
      cout << "query_cell: " << query_cell.transpose()
           << ", query_pt: " << query_pt.transpose() << endl;
      cout << actual_value << endl;
      cout << potential_value << endl;
    }
    EXPECT_NEAR(actual_value, potential_value, precision);
  }

 protected:
  double neighborhood_threshold_;
  double weight_threshold_;
  AnalyticalGrid analytical_grid_;
  std::shared_ptr<LinearMap> linear_function_;
  std::shared_ptr<LinearMap> constant_function_;
};
**/

TEST(TricubicInterpolation, Constant) {
  bool verbose = false;
  double delta = 1;
  double max = 30;
  uint32_t n = max / delta;
  uint32_t nb_points = n * n * n;
  Eigen::MatrixXd X(nb_points, 3);
  Eigen::VectorXd Y(nb_points);
  uint32_t i = 0;
  for (double z = 0.; z < max; z += delta) {
    for (double x = 0.; x < max; x += delta) {
      for (double y = 0.; y < max; y += delta) {
        X.row(i) << x, y, z;
        Y(i) = 17.;
        i++;
      }
    }
  }

  ASSERT_EQ(Y.size(), X.rows());

  std::vector<double> data(Y.data(), Y.data() + Y.size());
  auto tricubic = std::make_shared<TriCubicGridInterpolator>(data, 1, n, n, n);
  double precision = 1e-3;
  for (i = 0; i < X.rows(); i++) {
    double actual_value = 17.;
    double potential_value = tricubic->Evaluate(X.row(i));
    if (verbose) {
      cout << "------------------------" << endl;
      cout << "p: " << X.row(i) << endl;
      cout << "actual_value: " << actual_value << endl;
      cout << "potential_value: " << potential_value << endl;
    }
    EXPECT_NEAR(actual_value, potential_value, precision);
  }
}

TEST(TricubicInterpolation, Linear) {
  bool verbose = false;
  double delta = 1;
  double max = 30;
  uint32_t n = max / delta;
  uint32_t nb_points = n * n * n;
  Eigen::MatrixXd X(nb_points, 3);
  Eigen::VectorXd Y(nb_points);
  auto linear_function = SetUpLinearFunction();
  uint32_t i = 0;
  for (double z = 0.; z < max; z += delta) {
    for (double y = 0.; y < max; y += delta) {
      for (double x = 0.; x < max; x += delta) {
        X.row(i) << x, y, z;
        Y(i) = (*linear_function)(X.row(i))[0];
        i++;
      }
    }
  }

  ASSERT_EQ(Y.size(), X.rows());

  std::vector<double> data(Y.data(), Y.data() + Y.size());
  auto tricubic = std::make_shared<TriCubicGridInterpolator>(data, 1, n, n, n);
  double precision = 1e-3;
  for (i = 0; i < X.rows(); i++) {
    double actual_value = Y(i);
    double potential_value = tricubic->Evaluate(X.row(i));
    if (verbose) {
      cout << "------------------------" << endl;
      cout << "p: " << X.row(i) << endl;
      cout << "actual_value: " << actual_value << endl;
      cout << "potential_value: " << potential_value << endl;
    }
    EXPECT_NEAR(actual_value, potential_value, precision);
  }
}
