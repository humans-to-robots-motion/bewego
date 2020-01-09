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
static const unsigned int SEED = 1;

TEST(rotations, QuatToEuler_fd) {
  std::srand(SEED);

  f = std::make_shared<QuatToEuler>();
  ASSERT_TRUE(f->CheckJacobian());

}

TEST(rotations, QuatToEuler) {
  f = std::make_shared<QuatToEuler>();
  Eigen::Vector4d quat(0.64434867, -0.83761646, -0.13003315, -0.79692024);
  cout << f->Forward(quat) << endl;
  cout << f->Jacobian(quat) << endl;
  //ASSERT_TRUE(f->CheckJacobian());

}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
