/*
 * Copyright (c) 2021
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
 *                                                             Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/util/misc.h>
#include <bewego/util/util.h>

using bewego::util::range;

namespace bewego {

/*!\brief Base class to implement a function network
        It registers functions and evaluates
        f(x_0, x_1, ... ) = \sum_i f_i(x_0, x_1, ...)
 */
class FunctionNetwork : public DifferentiableMap {
 public:
  FunctionNetwork() {}
  virtual uint32_t output_dimension() const { return 1; }
  virtual uint32_t input_dimension() const = 0;

  /** Should return an array or single value */
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    Eigen::VectorXd value = Eigen::VectorXd::Zero(1);
    for (auto f : functions_) {
      value += f->Forward(x);
    }
    return value;
  }

  void AddFunction(DifferentiableMapPtr f) { functions_.push_back(f); }

 protected:
  std::vector<DifferentiableMapPtr> functions_;
};

/*!\brief Base class to implement a function network
        It allows to register functions and evaluates
        f(x_{i-1}, x_i, x_{i+1}) = \sum_i f_i(x_{i-1}, x_i, x_{i+1})
 */
class CliquesFunctionNetwork : public FunctionNetwork {
 public:
  CliquesFunctionNetwork(uint32_t input_dimension, uint32_t clique_element_dim)
      : FunctionNetwork(),
        input_size_(input_dimension),
        nb_clique_elements_(3),
        clique_element_dim_(clique_element_dim),
        clique_dim_(nb_clique_elements_ * clique_element_dim_),
        nb_cliques_(uint32_t(input_size_ / clique_element_dim) - 2),
        nb_terms_(0) {
    functions_.resize(nb_cliques_);
    for (uint32_t t = 0; t < nb_cliques_; t++) {
      functions_[t] = std::make_shared<SumMap>();
    }
  }

  virtual uint32_t input_dimension() const { return input_size_; }
  virtual uint32_t nb_cliques() const { return nb_cliques_; }
  virtual uint32_t n() const { return clique_element_dim_; }
  virtual uint32_t T() const { return nb_cliques_; }

  /** We call over all subfunctions in each clique */
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    Eigen::VectorXd value = Eigen::VectorXd::Zero(1);
    auto cliques = AllCliques(x);
    for (uint32_t t = 0; t < nb_cliques_; t++) {
      const Eigen::VectorXd& x_t = cliques[t];
      for (const auto& f : clique_functions(t)) {
        value += f->Forward(x_t);
      }
    }
    return value;
  }

  /**
  The jacboian matrix is of dimension m x n
              m (rows) : output size
              n (cols) : input size
          which can also be viewed becase the first order Taylor expansion
          of any differentiable map is f(x) = f(x_0) + J(x_0)_f x,
          where x is a collumn vector.
          The sub jacobian of the maps are the sum of clique jacobians
          each clique function f : R^dim -> R, where dim is the clique size.
  **/
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd J(
        Eigen::MatrixXd::Zero(output_dimension(), input_dimension()));
    auto cliques = AllCliques(x);
    for (uint32_t t = 0; t < nb_cliques_; t++) {
      const Eigen::VectorXd& x_t = cliques[t];
      for (const auto& f : clique_functions(t)) {
        assert(f->output_dimension() == output_dimension());
        uint32_t c_id = t * clique_element_dim_;
        J.block(0, c_id, 1, clique_dim_) += f->Jacobian(x_t);
      }
    }
    return J;
  }

  /**
  The hessian matrix is of dimension m x m
              m (rows) : input size
              m (cols) : input size
  **/
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd H(
        Eigen::MatrixXd::Zero(input_dimension(), input_dimension()));
    auto cliques = AllCliques(x);
    for (uint32_t t = 0; t < nb_cliques_; t++) {
      const Eigen::VectorXd& x_t = cliques[t];
      for (const auto& f : clique_functions(t)) {
        assert(f->output_dimension() == output_dimension());
        uint32_t c_id = t * clique_element_dim_;
        H.block(c_id, c_id, clique_dim_, clique_dim_) += f->Hessian(x_t);
      }
    }
    return H;
  }

  /**
     return the clique value
     TODO create a test using this function.
  **/
  virtual Eigen::VectorXd CliqueValue(uint32_t t,
                                      const Eigen::VectorXd& x_t) const {
    Eigen::VectorXd value = Eigen::VectorXd::Zero(1);
    for (const auto& f : clique_functions(t)) {
      value += f->Forward(x_t);
    }
    return value;
  }

  /**
      return the clique jacobian
      J : the full jacobian
   **/
  virtual Eigen::MatrixXd CliqueJacobian(uint32_t t,
                                         const Eigen::MatrixXd& J) const {
    uint32_t c_id = t * clique_element_dim_;
    return J.block(0, c_id, 1, clique_dim_);
  }

  /**
      return the clique hessian
      H : the full hessian
   **/
  virtual Eigen::MatrixXd CliqueHessian(uint32_t t,
                                        const Eigen::MatrixXd& H) const {
    uint32_t c_id = c_id = t * clique_element_dim_;
    return H.block(c_id, c_id, clique_dim_, clique_dim_);
  }

  //! returns the element of the input vector corresponding to a clique
  Eigen::VectorXd Clique(uint32_t t, const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return x.segment(t * clique_element_dim_, clique_dim_);
  }

  //! returns a list of all cliques
  std::vector<Eigen::VectorXd> AllCliques(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    assert(input_dimension() == (nb_cliques_ + 2) * clique_element_dim_);
    std::vector<Eigen::VectorXd> cliques(nb_cliques_);
    for (uint32_t t = 0; t < nb_cliques_; t++) {
      cliques[t] = Clique(t, x);
    }
    return cliques;
  }

  // Register function f for clique i
  void RegisterFunctionForClique(uint32_t t, DifferentiableMapPtr f) {
    assert(f->input_dimension() == clique_dim_);
    std::vector<DifferentiableMapPtr> functions = clique_functions(t);
    functions.push_back(f);
    auto functions_copy = std::make_shared<VectorOfMaps>(functions);
    functions_[t] = std::make_shared<SumMap>(functions_copy);
    nb_terms_++;
  }

  // Register function f
  void RegisterFunctionForAllCliques(DifferentiableMapPtr f) {
    assert(f->input_dimension() == clique_dim_);
    for (uint32_t t = 0; t < nb_cliques_; t++) {
      RegisterFunctionForClique(t, f);
    }
  }

  // Register function f
  void RegisterFunctionForLastClique(DifferentiableMapPtr f) {
    assert(f->input_dimension() == clique_dim_);
    uint32_t T = nb_cliques_ - 1;
    RegisterFunctionForClique(T, f);
  }

  // x_{t}
  DifferentiableMapPtr CenterOfCliqueMap() const {
    const uint32_t& n = clique_element_dim_;
    const uint32_t& d = nb_clique_elements_;
    return std::make_shared<RangeSubspaceMap>(n * d, range(n, (d - 1) * n));
  }

  // x_{t+1}
  DifferentiableMapPtr RightMostOfCliqueMap() const {
    const uint32_t& n = clique_element_dim_;
    const uint32_t& d = nb_clique_elements_;
    return std::make_shared<RangeSubspaceMap>(n * d, range((d - 1) * n, d * n));
  }

  // x_{t} ; x_{t+1}
  DifferentiableMapPtr RightOfCliqueMap() const {
    const uint32_t& n = clique_element_dim_;
    const uint32_t& d = nb_clique_elements_;
    return std::make_shared<RangeSubspaceMap>(n * d, range(n, d * n));
  }

  // x_{t-1}
  DifferentiableMapPtr LeftMostOfCliqueMap() const {
    const uint32_t& n = clique_element_dim_;
    const uint32_t& d = nb_clique_elements_;
    return std::make_shared<RangeSubspaceMap>(n * d, range(n));
  }

  // x_{t-1} ; x_{t}
  DifferentiableMapPtr LeftOfCliqueMap() const {
    const uint32_t& n = clique_element_dim_;
    const uint32_t& d = nb_clique_elements_;
    return std::make_shared<RangeSubspaceMap>(n * d, range((d - 1) * n));
  }

  const VectorOfMaps& clique_functions(uint32_t t) const {
    assert(t < nb_cliques_);
    return std::static_pointer_cast<const SumMap>(functions_[t])->terms();
  }

  uint32_t nb_terms() const { return nb_terms_; }
  uint32_t clique_dim() const { return clique_dim_; }

 private:
  uint32_t input_size_;
  uint32_t nb_clique_elements_;
  uint32_t clique_element_dim_;
  uint32_t clique_dim_;
  uint32_t nb_cliques_;
  uint32_t nb_terms_;
};

/**
    Wraps the active part of the Clique Function Network

        The idea is that the finite differences are approximated using
        cliques so for the first clique, these values are quite off.
        Since the first configuration is part of the problem statement
        we can take it out of the optimization and throw away the gradient
        computed for that configuration.

        TODO Test...
*/
class TrajectoryObjectiveFunction : public DifferentiableMap {
 public:
  TrajectoryObjectiveFunction(
      const Eigen::VectorXd& q_init,
      std::shared_ptr<const CliquesFunctionNetwork> function_network)
      : q_init_(q_init),
        n_(q_init.size()),
        function_network_(function_network) {}

  Eigen::VectorXd FullVector(const Eigen::VectorXd& x_active) const {
    assert(x_active.size() == input_dimension());
    Eigen::VectorXd x_full(function_network_->input_dimension());
    x_full.head(n_) = q_init_;
    x_full.segment(n_, input_dimension()) = x_active;
    return x_full;
  }

  uint32_t output_dimension() const {
    return function_network_->output_dimension();
  }

  uint32_t input_dimension() const {
    return function_network_->input_dimension() - n_;
  }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return Eigen::VectorXd::Constant(
        1, std::min(1e100, (*function_network_)(FullVector(x))[0]));
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd J = function_network_->Jacobian(FullVector(x));
    return J.block(0, n_, 1, input_dimension());
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd H = function_network_->Hessian(FullVector(x));
    return H.block(n_, n_, input_dimension(), input_dimension());
  }

  /** Hessian Sparcity Patern (band diagonal)

  In trajectory networks, the hessian is band diagonal.
  The Hessian matrix is a symmetric matrix, since the hypothesis of continuity
  of the second derivatives implies that the order of differentiation does not
  matter (Schwarz's theorem).
  **/
  util::MatrixSparsityPatern HessianSparcityPatern() const {
    util::MatrixSparsityPatern patern;
    uint32_t clique_dim = function_network_->clique_dim();
    for (uint32_t diag = 0; diag < clique_dim; diag++) {
      uint32_t i = diag;
      uint32_t j = 0;
      while (i < input_dimension() && j < input_dimension()) {
        if (diag == 0) {
          // main diagonal case
          patern.add_coefficient(i, i);
        } else {
          // other diagonals
          patern.add_coefficient(i, j);
          patern.add_coefficient(j, i);
          j++;
        }
        i++;
      }
    }
    return patern;
  }

 protected:
  Eigen::VectorXd q_init_;
  uint32_t n_;
  std::shared_ptr<const CliquesFunctionNetwork> function_network_;
};  // namespace bewego

/**
        Implement a trajectory as a single vector of configuration,
        returns cliques of configurations
        Note there is T active configuration in the trajectory
        indices
                0 and T + 1
            are supposed to be inactive.

        TODO: TEST...
*/
class Trajectory {
 public:
  Trajectory() : n_(0), T_(0) { x_ = Eigen::VectorXd(2); }
  Trajectory(uint32_t n, uint32_t T) : n_(n), T_(T) {
    assert(n_ > 0);
    assert(T_ > 0);
    x_ = Eigen::VectorXd(n_ * (T_ + 2));
  }

  Trajectory(const Eigen::VectorXd& q_init, uint32_t T) {
    assert(q_init.size() > 0);
    n_ = q_init.size();
    T_ = T;
    x_ = Eigen::VectorXd::Zero(n_ * (T_ + 2));
    x_.head(n_) = q_init;
    // x_.tail(n_ * (T_ + 1)) = x; // TODO
  }

  Trajectory(const Eigen::VectorXd& q_init, const Eigen::VectorXd& x) {
    assert(n > 0);
    assert(x.size() % q_init.size() == 0);
    n_ = q_init.size();
    T_ = uint32_t((x.size() / q_init.size()) - 1);
    x_ = Eigen::VectorXd::Zero(n_ * (T_ + 2));
    x_.head(n_) = q_init;
    x_.tail(n_ * (T_ + 1)) = x;
  }

  Trajectory(uint32_t n, uint32_t T, const Eigen::VectorXd& q_init,
             const Eigen::VectorXd& x)
      : Trajectory(q_init, x) {
    assert(n == q_init.size());
    assert(T == uint32_t((x.size() / q_init.size()) - 1));
  }

  friend std::ostream& operator<<(std::ostream& os, const Trajectory& traj) {
    os << " - n : " << traj.n() << "\n"
       << " - T : " << traj.T() << "\n"
       << " - x : \n"
       << traj.x() << "\n"
       << " - x.size : " << traj.x().size();
    return os;
  }

  uint32_t n() const { return n_; }
  uint32_t T() const { return T_; }
  Eigen::VectorXd x() const { return x_; }

  void set(const Eigen::VectorXd& x) {
    assert(x.size() == n_ * (T_ + 2));
    x_ = x;
  }

  /** The active segment of the trajectory
      removes the first configuration on the trajectory */
  Eigen::VectorXd ActiveSegment() const { return x_.tail(n_ * (T_ + 1)); }

  /** first configuration */
  Eigen::VectorXd InitialConfiguration() const { return Configuration(0); }

  /** last active configuration */
  Eigen::VectorXd FinalConfiguration() const { return Configuration(T_); }

  /** mutable configuration */
  Eigen::VectorBlock<Eigen::VectorXd> Configuration(uint32_t i) {
    assert(i >= 0 && i <= (T_ + 1));
    return Eigen::VectorBlock<Eigen::VectorXd>(x_.derived(), i * n_, n_);
  }

  /** non-mutable configuration */
  const Eigen::VectorBlock<const Eigen::VectorXd> Configuration(
      uint32_t i) const {
    assert(i >= 0 && i <= (T_ + 1));
    return Eigen::VectorBlock<const Eigen::VectorXd>(x_.derived(), i * n_, n_);
  }
  /**
  returns velocity at index i
          WARNING It is not the same convention as for the clique
                  Here we suppose the velocity at q_init to be 0,
                  so the finite difference is left sided (q_t - q_t-1)/dt
                  This is different from the right sided version
                  (q_t+1 - q_t)/dt implemented in the cost term module.

          With left side FD we directly get the integration scheme:

              q_{t+1} = q_t + v_t * dt + a_t * dt^2

          where v_t and a_t are velocity and acceleration
          at index t, with v_0 = 0. */
  Eigen::VectorXd Velocity(uint32_t i, double dt) const {
    if (i == 0) {
      return Eigen::VectorXd::Zero(n_);
    }
    auto q_i_1 = Configuration(i - 1);
    auto q_i_2 = Configuration(i);
    return (q_i_2 - q_i_1) / dt;
  }

  /** Returns acceleration at index i
          Note that we suppose velocity at q_init to be 0 */
  Eigen::VectorXd Acceleration(uint32_t i, double dt) const {
    uint32_t id_init = 0 ? i == 0 : i - 1;
    auto q_i_0 = Configuration(id_init);
    auto q_i_1 = Configuration(i);
    auto q_i_2 = Configuration(i + 1);
    return (q_i_2 - 2 * q_i_1 + q_i_0) / (dt * dt);
  }

  /** Return a tuple of configuration and velocity at index i */
  Eigen::VectorXd State(uint32_t i, double dt) const {
    auto q_t = Configuration(i);
    auto v_t = Velocity(i, dt);
    Eigen::VectorXd s_t(2 * n_);
    s_t.head(n_) = q_t;
    s_t.tail(n_) = v_t;
    return s_t;
  }

  /** Returns a clique of 3 configurations */
  Eigen::VectorXd Clique(uint32_t i) const {
    assert(i >= 1 && i <= T_);
    return x_.segment(n_ * (i - 1), 3 * n_);
  }

  /** Returns a list of configurations */
  std::vector<Eigen::VectorXd> Configurations() const {
    uint32_t nb_config = T_ + 1;
    std::vector<Eigen::VectorXd> line(nb_config);
    for (uint32_t t = 0; t < nb_config; t++) {
      line[t] = Configuration(t);
    }
    return line;
  }

  //! Returns the trajectory in the form of matrix
  //! Each row is a configuration
  Eigen::MatrixXd Matrix() const {
    Eigen::MatrixXd Xi(T_ + 2, n_);
    for (uint32_t t = 0; t <= T() + 1; t++) {
      Xi.row(t) = Configuration(t);
    }
    return Xi;
  }

 protected:
  uint32_t n_;
  uint32_t T_;
  Eigen::VectorXd x_;
};

//! Implements a trajectory that can be continously interpolated
class ContinuousTrajectory : public Trajectory {
 public:
  ContinuousTrajectory(const Trajectory& trajectory) : Trajectory(trajectory) {}

  // The trajectory is indexed by s \in [0, 1]
  Eigen::VectorXd ConfigurationAtParameter(double s) const {
    double l = length();
    assert(l > 0.);
    if (l == 0.) {
      return Configuration(0);
    }
    double d_param = s * l;
    Eigen::VectorXd q_prev = Configuration(0);
    double dist = 0.;
    for (uint32_t i = 1; i <= T(); i++) {
      Eigen::VectorXd q_curr = Configuration(i);
      double d = (q_curr - q_prev).norm();
      if (d_param <= (d + dist)) {
        return interpolate(q_prev, q_curr, d_param - dist, d);
      }
      dist += d;
      q_prev = q_curr;
    }
    return Eigen::VectorXd();
  }

  // Length in configuration space
  double length() const {
    double length = 0.;
    Eigen::VectorXd q_prev = Configuration(0);
    for (uint32_t i = 1; i <= T(); i++) {
      Eigen::VectorXd q_curr = Configuration(i);
      length += (q_curr - q_prev).norm();
      q_prev = q_curr;
    }
    return length;
  }

  Eigen::VectorXd interpolate(const Eigen::VectorXd& q_1,
                              const Eigen::VectorXd& q_2, double d_param,
                              double dist) const {
    // interpolate between configurations """
    //#assert d_param / dist <= 1.
    double alpha = std::min(d_param / dist, 1.);
    // assert alpha >= 0 and alpha <= 1., "alpha : {}".format(alpha)
    return (1. - alpha) * q_1 + alpha * q_2;
  }
};

//! Initialize a zero motion trajectory
std::shared_ptr<Trajectory> InitializeZeroTrajectory(
    const Eigen::VectorXd& q_init, uint32_t T);

//! Convert vector representation in Trajectory.
Trajectory GetTrajectory(const std::vector<Eigen::VectorXd>& line);

//! Resamples a trajectory to be of T unit of time
Trajectory Resample(const Trajectory& trajectory, uint32_t T);

//! Linear interpolated trajectory
Trajectory GetLinearInterpolation(const Eigen::VectorXd& q_init,
                                  const Eigen::VectorXd& q_goal, uint32_t T);

//! TODO
//! Get the precision matrix using the hessian of the control costs
//! sets constant scalars for the velocity and acceleration costs
// Eigen::MatrixXd GetControlCostPrecisionMatrix(uint32_t T, double dt,
//                                               uint32_t n);

//! TODO
//! Samples trajectories with 0 mean
// std::vector<Trajectory> SampleTrajectoriesZeroMean(uint32_t T, double dt,
//                                                    uint32_t n,
//                                                    uint32_t nb_samples,
//                                                    double std_dev = 1.5e-02);

//! check if trajectory is Null motion
bool IsTrajectoryNullMotion(const Trajectory& trajectory, double error = 0);

//! Trajectory collision checking.
bool DoesTrajectoryCollide(
    const Trajectory& trajectory,
    std::shared_ptr<const DifferentiableMap> collision_check,
    double margin = 0.);

//! Checks the equality of two markov trajectories
bool AreTrajectoriesEqual(const Trajectory& traj1, const Trajectory& traj2,
                          double error = 0);

//! Returns the trajectory in the form of matrix
//! Each row is a configuration
Trajectory InitializeFromMatrix(const Eigen::MatrixXd& matrix);

}  // namespace bewego
