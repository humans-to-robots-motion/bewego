// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/combination_operators.h>
#include <bewego/util/misc.h>
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

TEST(differentiable_map, equality) {
  std::srand(SEED);

  Eigen::VectorXd v1 = util::Random(3);
  Eigen::VectorXd v2 = util::Random(3);
  Eigen::VectorXd v3 = v1;
  Eigen::VectorXd v4 = v2;

  ASSERT_TRUE(v1 == v3);
  ASSERT_TRUE(v2 == v4);
  ASSERT_FALSE(v1 == v2);
}

TEST(atomic_operators, zero_map) {
  std::srand(SEED);

  f = std::make_shared<ZeroMap>(5, 5);
  ASSERT_TRUE(f->CheckJacobian());

  f = std::make_shared<ZeroMap>(1, 5);
  ASSERT_TRUE(f->CheckHessian());

  ASSERT_TRUE(f->type() == "ZeroMap");
}

TEST(atomic_operators, identity_map) {
  std::srand(SEED);

  f = std::make_shared<IdentityMap>(5);
  ASSERT_TRUE(f->CheckJacobian());

  f = std::make_shared<IdentityMap>(1);
  ASSERT_TRUE(f->CheckHessian());

  ASSERT_TRUE(f->type() == "IdentityMap");
}

TEST(atomic_operators, constant_map) {
  std::srand(SEED);

  Eigen::VectorXd a_1 = Eigen::VectorXd::Random(3);
  f = std::make_shared<ConstantMap>(5, a_1);
  ASSERT_TRUE(f->CheckJacobian());

  Eigen::VectorXd a_2 = Eigen::VectorXd::Random(1);
  f = std::make_shared<ConstantMap>(3, a_2);
  ASSERT_TRUE(f->CheckHessian());

  ASSERT_TRUE(f->type() == "ConstantMap");
}

TEST(atomic_operators, affine_map) {
  std::srand(SEED);
  const double precision = 1e-10;
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Random(3);
    f = std::make_shared<AffineMap>(a, b);
    ASSERT_TRUE(f->CheckJacobian(precision));

    a = Eigen::MatrixXd::Random(1, 2);
    b = Eigen::VectorXd::Random(1);
    f = std::make_shared<AffineMap>(a, b);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));

    auto a_1 = Eigen::VectorXd::Random(2);
    auto b_1 = util::Rand() + .5;
    f = std::make_shared<AffineMap>(a_1, b_1);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));

    double a_2 = util::Rand() + .5;
    double b_2 = util::Rand() + .5;
    f = std::make_shared<AffineMap>(a_2, b_2);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }

  ASSERT_TRUE(f->type() == "AffineMap");
}

TEST_F(DifferentialMapTest, linear_map) {
  std::srand(SEED);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 2);
  Eigen::VectorXd a_1 = Eigen::VectorXd::Random(3);
  double a_2 = util::Rand();
  auto phi1 = std::make_shared<LinearMap>(A);
  auto phi2 = std::make_shared<LinearMap>(a_1.transpose());
  auto phi3 = std::make_shared<LinearMap>(a_2);
  AddRandomTests(phi1, NB_TESTS);
  AddRandomTests(phi2, NB_TESTS);
  AddRandomTests(phi3, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi1->type() == "LinearMap");
  ASSERT_TRUE(phi2->type() == "LinearMap");
  ASSERT_TRUE(phi3->type() == "LinearMap");
}

TEST(atomic_operators, quadric_map) {
  std::srand(SEED);
  const double precision = 1e-10;
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 3);
    Eigen::VectorXd b = Eigen::VectorXd::Random(3);
    Eigen::VectorXd c = Eigen::VectorXd::Random(1);
    f = std::make_shared<QuadricMap>(a, b, c);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }

  ASSERT_TRUE(f->type() == "QuadricMap");
}

TEST(atomic_operators, squared_norm) {
  std::srand(SEED);
  const double precision = 1e-10;
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Random(2);
    f = std::make_shared<SquaredNorm>(x0);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }

  ASSERT_TRUE(f->type() == "SquaredNorm");
}

TEST(atomic_operators, range_subspace_map) {
  std::srand(SEED);
  const double precision = 1e-10;
  uint32_t n = 10;
  std::vector<uint32_t> indices;

  indices = {2, 3, 7};  // output_dimension = 3
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    f = std::make_shared<RangeSubspaceMap>(n, indices);
    ASSERT_TRUE(f->CheckJacobian(precision));
  }

  indices = {5};  // output_dimension = 1
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    f = std::make_shared<RangeSubspaceMap>(n, indices);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }

  ASSERT_TRUE(f->type() == "RangeSubspaceMap");
}

TEST(atomic_operators, scale) {
  std::srand(SEED);
  const double precision = 1e-10;
  uint32_t n = 10;

  for (uint32_t i = 0; i < NB_TESTS; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    double c = util::Rand();
    auto g = std::make_shared<QuadricMap>(a, b, c);
    f = std::make_shared<Scale>(g, util::Rand());
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }

  ASSERT_TRUE(f->type() == "Scale");
}

TEST(atomic_operators, offset) {
  std::srand(SEED);
  const double precision = 1e-10;
  uint32_t n = 10;

  for (uint32_t i = 0; i < NB_TESTS; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    Eigen::VectorXd c = Eigen::VectorXd::Random(1);
    auto g = std::make_shared<QuadricMap>(a, b, c);
    f = std::make_shared<Offset>(g, Eigen::VectorXd::Random(1));
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }

  ASSERT_TRUE(f->type() == "Offset");
}

TEST(atomic_operators, sum_map) {
  std::srand(SEED);
  const double precision = 1e-10;
  auto maps = std::make_shared<VectorOfMaps>();
  for (uint32_t i = 0; i < 3; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Random(3);
    maps->push_back(std::make_shared<AffineMap>(a, b));
  }

  f = std::make_shared<SumMap>(maps);
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    ASSERT_TRUE(f->CheckJacobian(precision));
  }

  ASSERT_TRUE(f->type() == "SumMap");
}

TEST(atomic_operators, product_map) {
  std::srand(SEED);
  const double precision = 1e-6;
  auto maps = std::make_shared<VectorOfMaps>();
  for (uint32_t i = 0; i < 2; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(5, 5);
    Eigen::VectorXd b = Eigen::VectorXd::Random(5);
    Eigen::VectorXd c = Eigen::VectorXd::Random(1);
    maps->push_back(std::make_shared<QuadricMap>(a, b, c));
  }
  f = std::make_shared<ProductMap>((*maps)[0], (*maps)[1]);
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }

  ASSERT_TRUE(f->type() == "ProductMap");
}

TEST(atomic_operators, min) {
  std::srand(SEED);
  const double precision = 1e-6;
  VectorOfMaps maps;
  for (uint32_t i = 0; i < 2; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(5, 5);
    Eigen::VectorXd b = Eigen::VectorXd::Random(5);
    Eigen::VectorXd c = Eigen::VectorXd::Random(1);
    maps.push_back(std::make_shared<QuadricMap>(a, b, c));
  }
  auto min_map = std::make_shared<Min>(maps);
  EXPECT_EQ(min_map->maps().size(), maps.size());
  // for (uint32_t i = 0; i < NB_TESTS; i++) {
  //   ASSERT_TRUE(f->CheckJacobian(precision));
  //   ASSERT_TRUE(f->CheckHessian(precision));
  // }

  ASSERT_TRUE(min_map->type() == "Min");
}

TEST(atomic_operators, second_order_tayler_approx) {
  std::srand(SEED);
  const double precision = 1e-6;

  Eigen::MatrixXd a = Eigen::MatrixXd::Random(5, 5);
  Eigen::VectorXd b = Eigen::VectorXd::Random(5);
  Eigen::VectorXd c = Eigen::VectorXd::Random(1);
  Eigen::MatrixXd H = .5 * (a + a.transpose());
  auto map = std::make_shared<QuadricMap>(H, b, c);

  for (uint32_t i = 0; i < NB_TESTS; i++) {
    ASSERT_TRUE(map->CheckJacobian(precision));
    ASSERT_TRUE(map->CheckHessian(precision));
    Eigen::VectorXd x = Eigen::VectorXd::Random(5);
    auto approx = std::make_shared<SecondOrderTaylorApproximation>(*map, x);
    double v1 = approx->ForwardFunc(x);
    double v2 = map->ForwardFunc(x);
    Eigen::VectorXd g1 = approx->Gradient(x);
    Eigen::VectorXd g2 = map->Gradient(x);
    Eigen::MatrixXd h = approx->Hessian(x);
    ASSERT_LT(std::fabs(v1 - v2), 1e-6);
    ASSERT_LT((g1 - g2).cwiseAbs().maxCoeff(), 1e-6);
    ASSERT_LT((H - h).cwiseAbs().maxCoeff(), 1e-6);
    ASSERT_TRUE(approx->type() == "SecondOrderTaylorApproximation");
  }
}

TEST_F(DifferentialMapTest, trigo_test_map) {
  std::srand(SEED);
  set_verbose(true);
  set_precisions(1e-6, -1);
  auto phi = std::make_shared<TrigoTestMap>();
  Eigen::VectorXd x;
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    x = util::RandomVector(2, -10, +10);
    function_tests_.push_back(std::make_pair(phi, x));
  }
  RunAllTests();
  ASSERT_TRUE(phi->type() == "TrigoTestMap");
}

TEST_F(DifferentialMapTest, log_barrier) {
  std::srand(SEED);
  auto phi = std::make_shared<LogBarrier>();
  // set_verbose(true);
  Eigen::VectorXd x(1);
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    x[0] = util::Rand() + 1;  // between 0 and 1.
    function_tests_.push_back(std::make_pair(phi, x));
  }
  // AddRandomTests(phi, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi->type() == "LogBarrier");
}

TEST_F(DifferentialMapTest, log_barrier_with_approx) {
  std::srand(SEED);
  auto phi = std::make_shared<LogBarrierWithApprox>(5);
  // set_verbose(true);
  Eigen::VectorXd x(1);
  for (uint32_t i = 0; i < 100; i++) {
    x[0] = util::RandUniform(.1, 1.);
    function_tests_.push_back(std::make_pair(phi, x));
  }
  // AddRandomTests(phi, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi->type() == "LogBarrierWithApprox");
}

TEST_F(DifferentialMapTest, soft_norm) {
  std::srand(SEED);
  double alpha = .05;
  uint32_t dim = 5;
  auto phi1 = std::make_shared<SoftNorm>(alpha, dim);
  auto phi2 = std::make_shared<SoftNorm>(alpha, Eigen::VectorXd::Random(dim));
  AddRandomTests(phi1, NB_TESTS);
  AddRandomTests(phi2, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi1->type() == "SoftNorm");
}

TEST_F(DifferentialMapTest, log_sum_exp) {
  std::srand(SEED);
  auto phi1 = std::make_shared<LogSumExp>(10, 1);
  auto phi2 = std::make_shared<LogSumExp>(10, 10);
  AddRandomTests(phi1, NB_TESTS);
  AddRandomTests(phi2, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi1->type() == "LogSumExp");
}

TEST_F(DifferentialMapTest, combined_output_map) {
  std::srand(SEED);
  auto maps = std::make_shared<VectorOfMaps>();
  for (uint32_t i = 0; i < 4; i++) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(5, 5);
    Eigen::VectorXd b = Eigen::VectorXd::Random(5);
    Eigen::VectorXd c = Eigen::VectorXd::Random(1);
    maps->push_back(std::make_shared<QuadricMap>(A, b, c));
  }
  auto phi = std::make_shared<CombinedOutputMap>(*maps);
  AddRandomTests(phi, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi->type() == "CombinedOutputMap");
}

TEST_F(DifferentialMapTest, multi_eval_map) {
  std::srand(SEED);
  set_verbose(false);
  set_precisions(1e-5, -1);
  auto maps = std::make_shared<VectorOfMaps>();
  std::shared_ptr<MultiEvalMap> phi;

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 5);
  Eigen::VectorXd b = Eigen::VectorXd::Random(3);
  auto f1 = std::make_shared<AffineMap>(A, b);
  phi = std::make_shared<MultiEvalMap>(f1, 3);
  AddRandomTests(phi, NB_TESTS);

  auto f2 = std::make_shared<TrigoTestMap>();
  phi = std::make_shared<MultiEvalMap>(f2, 3);
  AddRandomTests(phi, NB_TESTS);

  RunAllTests();
  ASSERT_TRUE(phi->type() == "MultiEvalMap");
}

TEST_F(DifferentialMapTest, square_map) {
  std::srand(SEED);
  auto phi1 = std::make_shared<SquareMap>();
  AddRandomTests(phi1, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi1->type() == "SquareMap");
}

TEST_F(DifferentialMapTest, soft_dist) {
  std::srand(SEED);
  // verbose_ = true;
  // set_precisions(1e-6, 1e-5);
  auto dist = std::make_shared<SquaredNorm>(5);
  auto phi = std::make_shared<SoftDist>(dist);
  AddRandomTests(phi, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi->type() == "SoftDist");
}

TEST_F(DifferentialMapTest, logistic) {
  std::srand(SEED);
  // verbose_ = true;
  // set_precisions(1e-6, 1e-5);
  auto phi = std::make_shared<Logistic>(.23, .67, 3.32);
  AddRandomTests(phi, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi->type() == "Logistic");
}

TEST_F(DifferentialMapTest, arccos) {
  std::srand(SEED);
  auto phi = std::make_shared<Arccos>();
  // set_verbose(true);
  set_precisions(1e-5, 1e-4);
  Eigen::VectorXd x(1);
  for (uint32_t i = 0; i < 100; i++) {
    x[0] = util::RandUniform(-.95, .95);
    function_tests_.push_back(std::make_pair(phi, x));
  }
  RunAllTests();
  ASSERT_TRUE(phi->type() == "Arccos");
}

TEST_F(DifferentialMapTest, dotproduct) {
  std::srand(SEED);
  set_verbose(false);
  set_precisions(1e-5, 1e-5);  // TODO check hessian
  uint32_t n = 10;
  Eigen::VectorXd x(1);
  for (uint32_t i = 0; i < 10; i++) {
    Eigen::MatrixXd A1 = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd A2 = Eigen::MatrixXd::Random(n, n);
    // Make sure H is symetric (and positive definite)
    A1 = A1 * A1.transpose();
    A2 = A2 * A2.transpose();
    auto phi1 = std::make_shared<LinearMap>(A1);
    auto phi2 = std::make_shared<LinearMap>(A2);
    auto dot = std::make_shared<DotProduct>(phi1, phi2);
    Eigen::VectorXd x = util::Random(n);
    function_tests_.push_back(std::make_pair(dot, x));
    ASSERT_TRUE(dot->type() == "DotProduct");
  }
  RunAllTests();
}

TEST_F(DifferentialMapTest, activation_weights) {
  std::srand(SEED);
  set_verbose(false);
  set_precisions(1e-6, 1e-5);

  function_tests_.clear();
  uint32_t dim = 3;
  uint32_t nb_tests = 10;
  uint32_t nb_activations = 10;

  VectorOfMaps activations;
  VectorOfMaps functions;
  for (uint32_t i = 0; i < nb_activations; ++i) {
    Eigen::VectorXd p0 = util::Random(dim);
    activations.push_back(std::make_shared<SquaredNorm>(p0) - .1);
    Eigen::VectorXd H = Eigen::MatrixXd::Random(dim, dim);
    Eigen::VectorXd b = util::Random(dim);
    double c = util::Rand();
    functions.push_back(std::make_shared<QuadricMap>(H * H.transpose(), b, c));
  }
  auto f = std::make_shared<ActivationWeights>(activations, functions);

  for (uint32_t i = 0; i < nb_tests; ++i) {
    Eigen::VectorXd x = util::Random(dim);
    function_tests_.push_back(std::make_pair(f, x));
  }

  ASSERT_TRUE(f->type() == "ActivationWeights");
  RunAllTests();
}

TEST_F(DifferentialMapTest, smooth_transition_test) {
  set_verbose(false);
  set_precisions(1e-6, 1e-5);

  function_tests_.clear();
  uint32_t dim = 3;
  uint32_t nb_tests = 10;

  Eigen::VectorXd p0 = util::Random(dim);
  auto phi = std::make_shared<SquaredNorm>(p0) + .5;

  Eigen::MatrixXd H1, H2;
  Eigen::VectorXd b1, b2;
  double c1, c2;

  H1 = Eigen::MatrixXd::Random(dim, dim);
  H1 = H1 * H1.transpose();
  b1 = util::Random(dim);
  c1 = util::Rand();
  auto f1 = std::make_shared<QuadricMap>(H1, b1, c1);

  H2 = Eigen::MatrixXd::Random(dim, dim);
  H2 = H2 * H2.transpose();
  b2 = util::Random(dim);
  c2 = util::Rand();
  auto f2 = std::make_shared<QuadricMap>(H2, b2, c2);

  // Test points
  auto f = std::make_shared<SmoothTransition>(f1, f2, phi, 10);
  for (uint32_t i = 0; i < nb_tests; ++i) {
    Eigen::VectorXd x = util::Random(dim);
    function_tests_.push_back(std::make_pair(f, x));
  }

  ASSERT_TRUE(f->type() == "SmoothTransition");
  RunAllTests();
}

TEST(SmoothTransitionParameter, temperature_parameter) {
  double threshold = 0.97;
  for (uint32_t i = 0; i < 10; ++i) {
    double d = util::Rand();
    double k = SmoothTransition::TemperatureParameter(d, threshold);
    auto logistic = std::make_shared<Logistic>(k, 0, 1);
    EXPECT_NEAR(threshold, (*logistic)(d / 2), 1e-7);
    EXPECT_NEAR(1 - threshold, (*logistic)(-d / 2), 1e-7);
  }
}
