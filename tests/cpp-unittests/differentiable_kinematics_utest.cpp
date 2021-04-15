// Copyright (c) 2021, University of Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/motion/differentiable_kinematics.h>
#include <bewego/util/misc.h>

using namespace bewego;
using std::cout;
using std::endl;

Eigen::VectorXd SetupPosVel(const Eigen::Vector3d& p,
                            const Eigen::Vector3d& euler,
                            const Eigen::Vector3d& vel) {
  Eigen::Matrix3d rotation;
  rotation = Eigen::AngleAxisd(euler.x(), Eigen::Vector3d::UnitX()) *
             Eigen::AngleAxisd(euler.y(), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(euler.z(), Eigen::Vector3d::UnitZ());
  Eigen::VectorXd x(24);
  x.segment(0, 3) = p;                // position
  x.segment(3, 3) = rotation.col(0);  // x_axis
  x.segment(6, 3) = rotation.col(1);  // y_axis
  x.segment(9, 3) = rotation.col(2);  // z_axis
  x.segment(12, 3) = vel;             // d/dt pos
  return x;
}

class PositionMapTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    function_tests_.clear();
    auto map = std::make_shared<Position>(10);
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = util::Random(map->input_dimension());
      function_tests_.push_back(std::make_pair(map, x));
    }
  }
};

TEST_F(PositionMapTest, Evaluation) {
  set_verbose(false);
  gradient_precision_ = 1e-6;
  RunAllTests();
}

class VelocityMapTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    function_tests_.clear();
    auto map = std::make_shared<Velocity>(10);
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = util::Random(map->input_dimension());
      function_tests_.push_back(std::make_pair(map, x));
    }
  }
};

TEST_F(VelocityMapTest, Evaluation) {
  set_verbose(false);
  gradient_precision_ = 1e-6;
  RunAllTests();
}

TEST(ForwardKinematicsMap, VelocityInFrame) {
  Eigen::Vector3d vel(.1 * util::Random(3));
  Eigen::VectorXd x(
      SetupPosVel(util::Random(3), util::RandomVector(3, 0, 2 * M_PI), vel));
  auto map = std::make_shared<VelocityInFrame>();
  Eigen::Matrix3d rotation;
  rotation.col(0) = x.segment(3, 3);  // x_axis
  rotation.col(1) = x.segment(6, 3);  // y_axis
  rotation.col(2) = x.segment(9, 3);  // z_axis
  Eigen::VectorXd dx1 = (*map)(x);
  Eigen::VectorXd dx2 = rotation.inverse() * vel;
  cout << " - dx1 : " << dx1.transpose() << endl;
  cout << " - dx2 : " << dx2.transpose() << endl;
  ASSERT_LT(std::fabs((dx1 - dx2).norm()), 1.e-7);
}

TEST(ForwardKinematicsMap, PlanarTansform) {
  Eigen::Vector3d euler(0, 0, .340349);
  Eigen::Vector3d vel(cos(euler.z()), sin(euler.z()), 0);
  Eigen::VectorXd x(SetupPosVel(util::Random(3), euler, vel));
  auto map = std::make_shared<VelocityInFrame>();
  Eigen::Vector3d dx1 = (*map)(x);
  Eigen::Vector3d dx2(1, 0, 0);
  cout << " - dx1 : " << dx1.transpose() << endl;
  cout << " - dx2 : " << dx2.transpose() << endl;
  ASSERT_LT(std::fabs((dx1 - dx2).norm()), 1.e-7);
}

class VelocityInFrameMapTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    function_tests_.clear();
    auto map = std::make_shared<VelocityInFrame>();
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = util::Random(map->input_dimension());
      function_tests_.push_back(std::make_pair(map, x));
    }
  }
};

TEST_F(VelocityInFrameMapTest, Evaluation) {
  set_verbose(false);
  gradient_precision_ = 1e-6;
  RunAllTests();
}

class HomogeneousTransform2dMapTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    function_tests_.clear();
    auto map = std::make_shared<HomogeneousTransform2d>(util::Random(2));
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = util::Random(map->input_dimension());
      function_tests_.push_back(std::make_pair(map, x));
    }
  }
};

TEST_F(HomogeneousTransform2dMapTest, Evaluation) {
  set_verbose(false);
  gradient_precision_ = 1e-6;
  RunAllTests();
}

TEST(HomogeneousTransform3d, Forward) {
  bool verbose = false;
  double r_precision = 1e-5;
  Eigen::Matrix3d R, R_t;
  Eigen::Vector3d p0(0., 0., 0.);
  auto transform = std::make_shared<HomogeneousTransform3d>(p0);
  for (uint32_t i = 0; i < 10; i++) {
    Eigen::VectorXd q(2 * M_PI * util::Random(6));
    R = Eigen::AngleAxisd(q[3], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(q[4], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(q[5], Eigen::Vector3d::UnitX());
    R_t = transform->Transform(q).linear();
    if (verbose) {
      cout << '=' << endl;
      cout << q.transpose() << endl;
      cout << endl;
      cout << R << endl;
      cout << R_t << endl;
      cout << (R - R_t) << endl;
      cout << ((R - R_t).norm()) << endl;
    }
    EXPECT_NEAR((R - R_t).cwiseAbs().maxCoeff(), 0., r_precision);
  }
}

class HomogeneousTransform3dMapTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    function_tests_.clear();
    auto map = std::make_shared<HomogeneousTransform3d>(util::Random(3));
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = 2 * M_PI * util::Random(map->input_dimension());
      function_tests_.push_back(std::make_pair(map, x));
    }
  }
};

TEST_F(HomogeneousTransform3dMapTest, Evaluation) {
  set_verbose(false);
  gradient_precision_ = 1e-6;
  RunAllTests();
}