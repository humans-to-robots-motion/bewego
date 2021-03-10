// Copyright (c) 2019, University Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/util/misc.h>
#include <gtest/gtest.h>
#include <numerical_optimization/ipopt_problem.h>
#include <numerical_optimization/optimization_test_functions.h>

using namespace bewego;
using namespace bewego::util;
using namespace bewego::numerical_optimization;
using namespace Ipopt;
using namespace std;
using std::cout;
using std::endl;

// Use this solution validation for the linear one-sided quadratic inequality
// penalties. This solution isn't actually as good as the one found by the
// selective penalty term. (See below.)
void IpoptQPValidateSolution(const ConstrainedSolution& solution,
                             bool verbose) {
  Eigen::VectorXd sol(10), lm(7), value(1), inequality(5), equality(2);

  sol << -1.10833837, 2.71244381, 1.57379586, 1.34409748, -0.98735542,
      -1.9763331, -0.32983447, -0.11374623, 0.89995774, -2.38376993;

  value << 0.4938668057494411;

  inequality << 0.42007981, 2.79055742, 1.51886129, 4.42377241, 1.10013266;

  ExpectNear(solution.x(), sol, 2e-4, verbose);
  // ExpectNear(solution.lagrange_multipliers(), lm, 1e-4, verbose);
  cout << "objective: " << solution.objective_value() << endl;

  auto objective = Eigen::VectorXd::Constant(1, solution.objective_value());
  ExpectNear(objective, value, 1e-4, verbose);

  cout << "inequality constraint:"
       << solution.inequality_constraint_values().transpose() << endl;
  ExpectNear(solution.inequality_constraint_values(), inequality, 3e-4,
             verbose);

  cout << "eqality constraint:"
       << solution.equality_constraint_values().transpose() << endl;
  // ExpectNear(solution.equality_constraint_values(), equality, 1e-5, verbose);
}

TEST_F(GenericQuadricProgramTest, IpoptSimpleOptimizeQuadraticProgram) {
  bool verbose = true;
  set_verbose(verbose);

  IpoptOptimizer constrained_optimizer;
  constrained_optimizer.set_verbose(verbose);

  // constrained_optimizer.set_option("max_iter", 1);
  constrained_optimizer.set_option("derivative_test", "first-order");
  constrained_optimizer.set_option("derivative_test", "second-order");
  constrained_optimizer.set_option("derivative_test_tol", 1e-5);

  ConstrainedSolution solution =
      constrained_optimizer.Run(*nonlinear_problem_, x0_);
  EXPECT_TRUE(solution.success());
  if (solution.success()) {
    IpoptQPValidateSolution(solution, verbose);
  }
}

TEST_F(GenericQuadricalyConstrainedQuadricProgramTest,
       IpoptSimpleOptimizeQuadraticProgram) {
  bool verbose = true;
  set_verbose(verbose);

  IpoptOptimizer constrained_optimizer;
  constrained_optimizer.set_verbose(verbose);

  // constrained_optimizer.set_option("max_iter", 1);
  constrained_optimizer.set_option("derivative_test", "first-order");
  constrained_optimizer.set_option("derivative_test", "second-order");
  constrained_optimizer.set_option("derivative_test_tol", 1e-5);

  ConstrainedSolution solution =
      constrained_optimizer.Run(*nonlinear_problem_, x0_);
  EXPECT_TRUE(solution.success());
  // TODO validate solution
  //  if (solution.success()) {
  //    IpoptQPValidateSolution(solution, verbose);
  //  }
}

// int main(int argc, char** argv) {
//   testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
