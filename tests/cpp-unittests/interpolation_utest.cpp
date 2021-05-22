// Copyright (c) 2016, Max Planck Institute.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/util/cubic_interpolation.h>
#include <bewego/util/interpolation.h>
#include <bewego/util/misc.h>
#include <gtest/gtest.h>

#include <Eigen/LU>

using namespace bewego;
using std::cout;
using std::endl;

#define TEST_DELTA 0.3
#define TEST_MAX 1.0

static const unsigned int SEED = 0;

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

std::shared_ptr<LinearMap> SetUpLinearFunction(uint32_t n = 3) {
  Eigen::MatrixXd a(1, 3);
  a << 1, 2, 3;
  a.conservativeResize(1, n);
  return std::make_shared<LinearMap>(a);
}

std::shared_ptr<ConstantMap> SetUpConstantFunction(uint32_t n = 3) {
  Eigen::VectorXd a(1);
  a << 17;
  return std::make_shared<ConstantMap>(n, a);
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

TEST(CubicInterpolatorTest, Evaluation) {
  std::srand(SEED);
  bool verbose = false;
  double precision = 1e-12;
  uint32_t N = 15;   // nb of points
  double delta = 1;  // spacing
  Eigen::VectorXd Y = util::Random(N);
  std::vector<double> data(Y.data(), Y.data() + Y.size());
  auto cubic_inter = std::make_shared<CubicInterpolator>(data, delta);
  for (uint32_t i = 0; i < 10; i++) {
    double x = N * util::Rand();
    double dx = 0;
    auto p = cubic_inter->Neighboors(x, dx);
    double v1 = cubic_inter->Evaluate(x);
    double v2 = cubic_inter->Interpolate(p, dx);
    if (verbose) {
      cout << "------------------------" << endl;
      cout << "Y  : " << Y.transpose() << endl;
      cout << "x: " << x << endl;
      cout << "v1: " << v1 << endl;
      cout << "v2: " << v2 << endl;
      cout << "p : " << p[0] << " , " << p[1] << " , " << p[2] << " , " << p[3];
    }
    EXPECT_NEAR(v1, v2, precision);
  }
}

TEST(CubicInterpolatorTest, Derivative) {
  std::srand(SEED);
  bool verbose = false;
  double precision = 1e-06;
  uint32_t N = 15;   // nb of points
  double delta = 1;  // spacing
  Eigen::VectorXd Y = util::Random(N);
  std::vector<double> data(Y.data(), Y.data() + Y.size());
  auto cubic_inter = std::make_shared<CubicInterpolator>(data, delta);
  for (uint32_t i = 0; i < 10; i++) {
    double x = N * util::Rand();
    double dx = 1e-4;
    double dx_half = dx * .5;
    double v1 = cubic_inter->Evaluate(x - dx_half);
    double v2 = cubic_inter->Evaluate(x + dx_half);
    double dv1 = (v2 - v1) / dx;
    double dv2 = cubic_inter->Derivative(x);
    if (verbose) {
      cout << "------------------------" << endl;
      cout << "Y  : " << Y.transpose() << endl;
      cout << "x: " << x << endl;
      cout << "dv1: " << dv1 << endl;
      cout << "dv2: " << dv2 << endl;
    }
    EXPECT_NEAR(dv1, dv2, precision);
  }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

class BiCubicTest : public ::testing::Test {
 public:
  void SetUp() {
    verbose_ = false;
    linear_function_ = SetUpLinearFunction(2);
    constant_function_ = SetUpConstantFunction(2);
    delta_ = 1;
    max_ = 30;
    n_ = max_ / delta_;
  }

  void InitializeGrid(std::shared_ptr<DifferentiableMap> f) {
    uint32_t nb_points = n_ * n_;
    X_ = Eigen::MatrixXd(nb_points, 2);
    Y_ = Eigen::VectorXd(nb_points);
    ASSERT_EQ(Y_.size(), X_.rows());
    uint32_t i = 0;
    for (double x = 0.; x < max_; x += delta_) {
      for (double y = 0.; y < max_; y += delta_) {
        X_.row(i) << x, y;
        Y_(i) = (*f)(X_.row(i))[0];
        i++;
      }
    }
  }

  void ValidateSimpleGrid() {
    std::vector<double> data(Y_.data(), Y_.data() + Y_.size());
    auto bicubic = std::make_shared<BiCubicGridInterpolator>(data, 1, n_, n_);
    double precision = 1e-3;
    for (uint i = 0; i < X_.rows(); i++) {
      double actual_value = Y_(i);
      const Eigen::Vector2d &p = X_.row(i);
      double dx, dy;
      auto neigh = bicubic->Neighboors(p.x(), p.y(), dx, dy);
      double potential_value = bicubic->Interpolate(neigh, dx, dy);
      if (verbose_) {
        cout << "------------------------" << endl;
        cout << "p: " << X_.row(i) << endl;
        cout << "actual_value: " << actual_value << endl;
        cout << "potential_value: " << potential_value << endl;
      }
      ASSERT_NEAR(actual_value, potential_value, precision);
    }
  }

  void ValidateGrid() {
    std::vector<double> data(Y_.data(), Y_.data() + Y_.size());
    auto bicubic = std::make_shared<BiCubicGridInterpolator>(data, 1, n_, n_);
    double precision = 1e-3;
    for (uint i = 0; i < X_.rows(); i++) {
      const Eigen::Vector2d &p = X_.row(i);
      double dx, dy;
      auto neigh = bicubic->Neighboors(p.x(), p.y(), dx, dy);
      double actual_value = Y_(i);
      double potential_value1 = bicubic->Evaluate(p);
      double potential_value2 = bicubic->Interpolate(neigh, dx, dy);
      if (verbose_) {
        cout << "------------------------" << endl;
        cout << "p: " << X_.row(i) << endl;
        cout << "actual_value: " << actual_value << endl;
        cout << "potential_value1: " << potential_value1 << endl;
        cout << "potential_value1: " << potential_value2 << endl;
      }
      ASSERT_NEAR(actual_value, potential_value1, precision);
      ASSERT_NEAR(actual_value, potential_value2, precision);
    }
  }

  void ValidateGradientGrid() {
    std::vector<double> data(Y_.data(), Y_.data() + Y_.size());
    auto bicubic = std::make_shared<BiCubicGridInterpolator>(data, 1, n_, n_);
    double precision = 1e-3;  // TODO make that tighter?
    double dt = 1e-4;
    double dt_half = dt * .5;
    double v1, v2, dv_x, dv_y;
    Eigen::Vector2d p1, p2;
    for (uint i = 0; i < X_.rows(); i++) {
      const Eigen::Vector2d &p = X_.row(i);

      // Finite difference
      p1 << p.x() - dt_half, p.y();
      p2 << p.x() + dt_half, p.y();
      v1 = bicubic->Evaluate(p1);
      v2 = bicubic->Evaluate(p2);
      dv_x = (v2 - v1) / dt;
      p1 << p.x(), p.y() - dt_half;
      p2 << p.x(), p.y() + dt_half;
      v1 = bicubic->Evaluate(p1);
      v2 = bicubic->Evaluate(p2);
      dv_y = (v2 - v1) / dt;

      // Analytic
      Eigen::Vector2d g = bicubic->Gradient(p);

      if (verbose_) {
        cout << "------------------------" << endl;
        cout << "p: " << p.transpose() << endl;
        cout << "g: " << g.transpose() << endl;
        cout << "dv_x: " << dv_x << endl;
        cout << "dv_x: " << dv_y << endl;
      }
      EXPECT_NEAR(dv_x, g.x(), precision);
      EXPECT_NEAR(dv_y, g.y(), precision);
    }
  }

  void PrintMatrix() const {
    typedef const Eigen::MatrixXd MatrixType;
    MatrixType grid = Eigen::Map<MatrixType>(Y_.data(), n_, n_);
    cout << " grid_ : " << endl << grid << endl;
  }

 protected:
  bool verbose_;
  double delta_;
  double max_;
  uint32_t n_;
  Eigen::MatrixXd X_;
  Eigen::VectorXd Y_;
  std::shared_ptr<DifferentiableMap> linear_function_;
  std::shared_ptr<DifferentiableMap> constant_function_;
};

TEST_F(BiCubicTest, Constant) {
  verbose_ = false;
  InitializeGrid(constant_function_);
  ValidateSimpleGrid();
}

TEST_F(BiCubicTest, LinearSimple) {
  verbose_ = false;
  InitializeGrid(linear_function_);
  ValidateSimpleGrid();
  if (verbose_) {
    PrintMatrix();
  }
}

TEST_F(BiCubicTest, Linear) {
  verbose_ = false;
  InitializeGrid(linear_function_);
  ValidateGrid();
}

TEST_F(BiCubicTest, Gradient) {
  verbose_ = false;
  InitializeGrid(linear_function_);
  ValidateGradientGrid();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

class TriCubicTest : public ::testing::Test {
 public:
  void SetUp() {
    verbose_ = false;
    linear_function_ = SetUpLinearFunction();
    constant_function_ = SetUpConstantFunction();
    delta_ = 1;
    max_ = 30;
    n_ = max_ / delta_;
  }

  void InitializeGrid(std::shared_ptr<DifferentiableMap> f) {
    uint32_t nb_points = n_ * n_ * n_;
    X_ = Eigen::MatrixXd(nb_points, 3);
    Y_ = Eigen::VectorXd(nb_points);
    ASSERT_EQ(Y_.size(), X_.rows());
    uint32_t i = 0;
    for (double z = 0.; z < max_; z += delta_) {
      for (double y = 0.; y < max_; y += delta_) {
        for (double x = 0.; x < max_; x += delta_) {
          X_.row(i) << x, y, z;
          Y_(i) = (*f)(X_.row(i))[0];
          i++;
        }
      }
    }
  }

  void ValidateGrid() {
    std::vector<double> data(Y_.data(), Y_.data() + Y_.size());
    auto tricubic =
        std::make_shared<TriCubicGridInterpolator>(data, 1, n_, n_, n_);
    double precision = 1e-3;
    for (uint i = 0; i < X_.rows(); i++) {
      double actual_value = Y_(i);
      double potential_value = tricubic->Evaluate(X_.row(i));
      if (verbose_) {
        cout << "------------------------" << endl;
        cout << "p: " << X_.row(i) << endl;
        cout << "actual_value: " << actual_value << endl;
        cout << "potential_value: " << potential_value << endl;
      }
      EXPECT_NEAR(actual_value, potential_value, precision);
    }
  }

 protected:
  bool verbose_;
  double delta_;
  double max_;
  uint32_t n_;
  Eigen::MatrixXd X_;
  Eigen::VectorXd Y_;
  std::shared_ptr<DifferentiableMap> linear_function_;
  std::shared_ptr<DifferentiableMap> constant_function_;
};

TEST_F(TriCubicTest, Constant) {
  verbose_ = false;
  InitializeGrid(constant_function_);
  ValidateGrid();
}

TEST_F(TriCubicTest, Linear) {
  verbose_ = false;
  InitializeGrid(linear_function_);
  ValidateGrid();
}
