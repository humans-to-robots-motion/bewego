/*
 * Copyright (c) 2019
 * All rights reserved.
 *
 * Redistribution  and  use  in  source  and binary  forms,  with  or  without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of  source  code must retain the  above copyright
 *      notice and this list of conditions.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice and  this list of  conditions in the  documentation and/or
 *      other materials provided with the distribution.
 *
 * THE SOFTWARE  IS PROVIDED "AS IS"  AND THE AUTHOR  DISCLAIMS ALL WARRANTIES
 * WITH  REGARD   TO  THIS  SOFTWARE  INCLUDING  ALL   IMPLIED  WARRANTIES  OF
 * MERCHANTABILITY AND  FITNESS.  IN NO EVENT  SHALL THE AUTHOR  BE LIABLE FOR
 * ANY  SPECIAL, DIRECT,  INDIRECT, OR  CONSEQUENTIAL DAMAGES  OR  ANY DAMAGES
 * WHATSOEVER  RESULTING FROM  LOSS OF  USE, DATA  OR PROFITS,  WHETHER  IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR  OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 *
 *                                              Jim Mainprice We 18 Dec 2019
 */

#include <bewego/numerical_optimization/ipopt_optimizer.h>
#include <bewego/numerical_optimization/ipopt_problem.h>
#include <bewego/util/misc.h>

using namespace Ipopt;
using namespace bewego;
using namespace bewego::numerical_optimization;
using std::cout;
using std::endl;

std::string ipopt_options = "";  // TODO
// DEFINE_string(ipopt_options, "first, 1, second, 2, third, 3", "");

static std::map<int, std::string> ErrorTypes;
//   {"Solve_Succeeded",                    // 0
//   "Solved_To_Acceptable_Level",          // 1
//   "Infeasible_Problem_Detected",         // 2
//   "Search_Direction_Becomes_Too_Small",  // 3
//   "Diverging_Iterates",                  // 4
//   "User_Requested_Stop",                 // 5
//   "Feasible_Point_Found",                // 6

//   "Maximum_Iterations_Exceeded",
//   "Restoration_Failed",
//   "Error_In_Step_Computation",
//   "Maximum_CpuTime_Exceeded",
//   "Not_Enough_Degrees_Of_Freedom",
//   "Invalid_Problem_Definition",
//   "Invalid_Option",
//   "Invalid_Number_Detected",

//   "Unrecoverable_Exception",
//   "NonIpopt_Exception_Thrown",
//   "Insufficient_Memory",
//   "Internal_Error"};

std::vector<std::pair<std::string, double>> ParseIpoptOptions() {
  std::string ipopt_options = "";  // TODO get that towork...
  auto options_str = util::ParseCsvString(ipopt_options);
  cout << "ipopt_options : " << ipopt_options << endl;
  cout << "options_str.size() : " << options_str.size() << endl;
  assert(options_str.size() % 2 != 1);
  std::vector<std::pair<std::string, double>> options;
  for (uint32_t i = 0; i < options_str.size(); i += 2) {
    auto name = options_str[i];
    auto value = std::stod(options_str[i + 1]);
    options.push_back(std::make_pair(name, value));
  }
  return options;
}

IpoptOptimizer::IpoptOptimizer() {
  ipopt_app_ = std::make_shared<Ipopt::IpoptApplication>();
  bounds_.clear();

  /* Which linear solver to use. Mumps is default because it comes with the
   * precompiled ubuntu binaries. However, the coin-hsl solvers can be
   * significantly faster and are free for academic purposes. They can be
   * downloaded here: http://www.hsl.rl.ac.uk/ipopt/ and must be compiled
   * into your IPOPT libraries. Then you can use the additional strings:
   * "ma27, ma57, ma77, ma86, ma97" here.
   */
  set_option("linear_solver", "mumps");

  /* whether to use the analytical derivatives "exact" coded in ifopt, or let
   * IPOPT approximate these through "finite-difference-values". This is usually
   * significantly slower.
   * For the hessian you can use : "limited-memory"
   */
  set_option("jacobian_approximation", "exact");
  set_option("hessian_approximation", "exact");
  set_option("max_cpu_time", 40.0);
  set_option("tol", 0.001);
  set_option("print_timing_statistics", "no");
  set_option("print_user_options", "no");
  set_option("print_level", 4);

  // See test execuables.
  // set_option("max_iter", 1);
  // set_option("derivative_test", "first-order");
  // set_option("derivative_test", "second-order");
  // set_option("derivative_test_tol", 1e-5);
}

void IpoptOptimizer::set_option(const std::string& name,
                                const std::string& value) {
  ipopt_app_->Options()->SetStringValue(name, value);
}

void IpoptOptimizer::set_option(const std::string& name, int value) {
  ipopt_app_->Options()->SetIntegerValue(name, value);
}

void IpoptOptimizer::set_option(const std::string& name, double value) {
  ipopt_app_->Options()->SetNumericValue(name, value);
}

double IpoptOptimizer::total_wallclock_time() const {
  return ipopt_app_->Statistics()->TotalWallclockTime();
}

// Reads flags to set options
// Parse all options from flags
void IpoptOptimizer::set_options_map(
    const std::map<std::string, double>& options) {
  // cout << "Parse ipopt options..." << endl;
  // auto options = ParseIpoptOptions(options);
  // cout << "options.size() : " << options.size() << endl;
  for (const auto& v : options) {
    cout << " -- set option : " << v.first << " , to : " << v.second << endl;
    set_option(v.first, v.second);
  }
}

std::vector<numerical_optimization::Bounds> IpoptOptimizer::GetVariableBounds(
    uint32_t n) const {
  if (bounds_.empty()) {
    // Generic bounds
    std::vector<numerical_optimization::Bounds> bounds(n);
    for (auto& bound : bounds) {
      bound.upper_ = std::numeric_limits<double>::max();
      bound.lower_ = std::numeric_limits<double>::lowest();
    }
    return bounds;
  }
  assert(bounds_.size() == n);
  return bounds_;
}

std::vector<numerical_optimization::Bounds>
IpoptOptimizer::GetGenericConstraintsBounds(uint32_t n_g, uint32_t n_h) const {
  std::vector<numerical_optimization::Bounds> bounds(n_g + n_h);
  for (uint32_t i = 0; i < n_g; i++) {
    bounds[i].upper_ = std::numeric_limits<double>::max();
  }
  return bounds;
}

ConstrainedSolution IpoptOptimizer::Run(
    const OptimizationProblemWithConstraints& problem,
    const Eigen::VectorXd& x0) const {
  using namespace Ipopt;

  int status = ipopt_app_->Initialize();
  if (status != Solve_Succeeded) {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    throw std::length_error("Ipopt could not initialize correctly");
  }

  // check the jacobian_approximation method
  std::string jac_type = "";
  ipopt_app_->Options()->GetStringValue("jacobian_approximation", jac_type, "");

  // convert the NLP problem to Ipopt
  IpoptProblem::NonLinearProblemPtr nlp =
      std::make_shared<const OptimizationProblemWithConstraints>(problem);
  SmartPtr<IpoptProblem> ipopt_problem = new IpoptProblem(
      nlp, GetVariableBounds(nlp->objective_function()->input_dimension()),
      GetGenericConstraintsBounds(nlp->num_inequality_constraints(),
                                  nlp->num_equality_constraints()),
      x0, jac_type == "finite-difference-values", publish_solution_);
  ipopt_problem->set_verbose(verbose_);

  bool success = true;
  status = ipopt_app_->OptimizeTNLP(ipopt_problem);
  if ((status != Solve_Succeeded) && (status != Solved_To_Acceptable_Level)) {
    std::string msg = "ERROR: Ipopt failed to find a solution. Return Code: " +
                      std::to_string(status) + "\n";
    std::cerr << msg;
    success = false;
  }
  Eigen::VectorXd g_evaluations;
  Eigen::VectorXd h_evaluations;
  auto x_solution = ipopt_problem->solution();
  auto objective_value =
      problem.Evaluate(x_solution, &g_evaluations, &h_evaluations);
  return ConstrainedSolution(
      x_solution, objective_value, g_evaluations, h_evaluations,
      (success) ? (ConstrainedSolution::NO_WARNING)
                : (ConstrainedSolution::DID_NOT_CONVERGE));
}
