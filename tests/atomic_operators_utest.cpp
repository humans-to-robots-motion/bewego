// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <gtest/gtest.h>
#include <bewego/atomic_operators.h>
#include<random>

using namespace bewego;
using std::cout;
using std::endl;

std::shared_ptr<DifferentiableMap> f;

TEST(atomic_operators, zero_map)
{
    std::srand((unsigned int) 0);
    
    f = std::make_shared<ZeroMap>(5, 5);
    ASSERT_TRUE(f->CheckJacobian());

    f = std::make_shared<ZeroMap>(1, 5);
    ASSERT_TRUE(f->CheckHessian());
}

TEST(atomic_operators, identity_map)
{
    std::srand((unsigned int) 0);

    f = std::make_shared<IdentityMap>(5);
    ASSERT_TRUE(f->CheckJacobian());

    f = std::make_shared<IdentityMap>(1);
    ASSERT_TRUE(f->CheckHessian());
}

TEST(atomic_operators, affine_map)
{
    std::srand((unsigned int) 0);

    Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Random(3);

    f = std::make_shared<AffineMap>(a, b);
        
    ASSERT_TRUE(f->CheckJacobian());

    a = Eigen::MatrixXd::Random(1, 2);
    b = Eigen::VectorXd::Random(1);

    f = std::make_shared<AffineMap>(a, b);
    
    ASSERT_TRUE(f->CheckJacobian());
    ASSERT_TRUE(f->CheckHessian());
}

TEST(atomic_operators, squared_norm)
{
    std::srand((unsigned int) 0);

    Eigen::VectorXd x0 = Eigen::VectorXd::Random(2);
    
    f = std::make_shared<SquaredNorm>(x0);

    ASSERT_TRUE(f->CheckJacobian());
    ASSERT_TRUE(f->CheckHessian());
}

int main(int argc, char* argv[]) {
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}