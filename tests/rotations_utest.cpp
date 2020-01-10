#include <bewego/rotations.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>

using namespace bewego;
using std::cout;
using std::endl;


std::shared_ptr<DifferentiableMap> f;
static const uint32_t NB_TESTS = 10;
static const unsigned int SEED = 0;

TEST(rotations, QuatToEulerCheckForward) {
  f = std::make_shared<QuatToEuler>();
  Eigen::Vector4d quat (0.48610611,  0.6118317,  -0.14720144, -0.65144289);
  Eigen::Vector3d correct (-1.83638469, -0.71290728,  1.3123043);
  auto res = f->Forward(quat);
  ASSERT_TRUE(res.isApprox(correct, 1e-7));
}

TEST(rotations, QuatToEulerCheckJacobian) {
  f = std::make_shared<QuatToEuler>();
  Eigen::Vector4d quat(-0.58377744, -0.4719759, 0.72638568, 0.04159109);
  Eigen::MatrixXd correct(3, 4);
  correct << 3.06874722,  2.16385246,  0.21609872,  0.26728813,-2.4705506, 0.14145777, 1.98551782, -1.60526342, 0.75672101, -0.91196898,  2.77735716, -1.16461729;
  auto jac = f->Jacobian(quat);
  ASSERT_TRUE(jac.isApprox(correct, 1e-7));
}

TEST(rotations, QuatToEulerCheckJacobianFD) {
  std::srand(SEED);
  f = std::make_shared<QuatToEuler>();
  ASSERT_TRUE(f->CheckJacobian(1e-7));
}


int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
