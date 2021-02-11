// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#pragma once

#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/util.h>

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
        nb_cliques_(uint32_t(input_size_ / clique_element_dim - 2)) {
    // TODO implement this as a Sum of maps
    functions_.resize(nb_cliques_);
  }

  virtual uint32_t input_dimension() const { return input_size_; }
  virtual uint32_t nb_cliques() const { return nb_cliques_; }

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
        Eigen::MatrixXd::Zero(output_dimension(), input_dimension()));
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

  // returns a list of all cliques
  std::vector<Eigen::VectorXd> AllCliques(const Eigen::VectorXd& x) const {
    std::vector<Eigen::VectorXd> cliques(nb_cliques_);
    for (uint32_t c_id = 0; c_id < clique_element_dim_ * nb_cliques_;
         c_id += clique_element_dim_) {
      cliques[c_id] = x.segment(c_id, clique_dim_);
    }
    assert(cliques.size() == nb_cliques_);
    return cliques;
  }

  // Register function f for clique i
  void RegisterFunctionForClique(uint32_t t, DifferentiableMapPtr f) {
    assert(f->input_dimension() == clique_dim_);
    std::vector<DifferentiableMapPtr> functions = clique_functions(t);
    functions.push_back(f);
    functions_[t] =
        std::make_shared<SumMap>(std::make_shared<VectorOfMaps>(functions));
  }

  // Register function f
  void RegisterFunctionForAllAliques(DifferentiableMapPtr f) {
    assert(f->input_dimension() == clique_dim_);
    for (uint32_t t = 0; t < nb_cliques_; t++) {
      RegisterFunctionForClique(t, f);
    }
  }

  // Register function f
  void RegisterFunctionLastClique(DifferentiableMapPtr f) {
    assert(f->input_dimension() == clique_dim_);
    uint32_t T = nb_cliques_ - 1;
    RegisterFunctionForClique(T, f);
  }

  // x_{t}
  DifferentiableMapPtr CenterOfCliqueMap() const {
    uint32_t dim = clique_element_dim_;
    return std::make_shared<RangeSubspaceMap>(
        dim * nb_clique_elements_,
        util::range(dim, (nb_clique_elements_ - 1) * dim));
  }

  // x_{t+1}
  DifferentiableMapPtr RightMostOfCliqueMap() const {
    uint32_t dim = clique_element_dim_;
    return std::make_shared<RangeSubspaceMap>(
        dim * nb_clique_elements_, util::range((nb_clique_elements_ - 1) * dim,
                                               nb_clique_elements_ * dim));
  }

  // x_{t} ; x_{t+1}
  DifferentiableMapPtr RightOfCliqueMap() const {
    uint32_t dim = clique_element_dim_;
    return std::make_shared<RangeSubspaceMap>(
        dim * clique_element_dim_, util::range(dim, nb_clique_elements_ * dim));
  }

  // x_{t-1}
  DifferentiableMapPtr LeftMostOfCliqueMap() const {
    uint32_t dim = clique_element_dim_;
    return std::make_shared<RangeSubspaceMap>(dim * nb_clique_elements_,
                                              util::range(dim));
  }

  // x_{t-1} ; x_{t}
  DifferentiableMapPtr LeftOfCliqueMap() {
    uint32_t dim = clique_element_dim_;
    return std::make_shared<RangeSubspaceMap>(
        dim * nb_clique_elements_,
        util::range((nb_clique_elements_ - 1) * dim));
  }

  const VectorOfMaps& clique_functions(uint32_t t) const {
    return std::static_pointer_cast<const SumMap>(functions_[t])->terms();
  }

 private:
  uint32_t input_size_;
  uint32_t nb_clique_elements_;
  uint32_t clique_element_dim_;
  uint32_t clique_dim_;
  uint32_t nb_cliques_;
};

/**
    Wraps the active part of the Clique Function Network

        The idea is that the finite differences are approximated using
        cliques so for the first clique, these values are quite off.
        Since the first configuration is part of the problem statement
        we can take it out of the optimization and through away the gradient
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
    Eigen::VectorXd value(1);
    value[0] = std::min(1e100, (*function_network_)(FullVector(x))[0]);
    return value;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd J = function_network_->Jacobian(FullVector(x));
    return J.block(0, n_, 1, input_dimension());
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd H = function_network_->Hessian(FullVector(x));
    return H.block(n_, n_, input_dimension(), input_dimension());
  }

 protected:
  Eigen::VectorXd q_init_;
  uint32_t n_;
  std::shared_ptr<const CliquesFunctionNetwork> function_network_;
};

/**
        Implement a trajectory as a single vector of configuration,
        returns cliques of configurations
        Note there is T active configuration in the trajectory
        indices
                0 and T + 1
            are supposed to be inactive.
*/
class Trajectory {
 public:
  Trajectory(uint32_t T, uint32_t n) : n_(n), T_(T) {
    assert(n_ > 0);
    assert(T_ > 0);
    x_ = Eigen::VectorXd(n_ * (T_ + 2));
  }

  Trajectory(uint32_t T, uint32_t n, const Eigen::VectorXd& q_init,
             const Eigen::VectorXd& x) {
    assert(n > 0);
    assert(x.size() % q_init.size() == 0);
    n_ = q_init.size();
    T_ = uint32_t((x.size() / q_init.size()) - 1);
    x_ = Eigen::VectorXd::Zero(n_ * (T_ + 2));
    x_.head(n_) = q_init;
    x_.tail(n_ * (T_ + 1)) = x;
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

  /**
  The active segment of the trajectory
      removes the first configuration on the trajectory */
  Eigen::VectorXd ActiveSegment() const { return x_.tail(n_ * (T_ + 1)); }

  /** first configuration */
  Eigen::VectorXd InitialConfiguration() const { return Configuration(0); }

  /** last active configuration */
  Eigen::VectorXd FinalConfiguration() const { return Configuration(T_); }

  /** mutable : traj.configuration(3)[:] = np.ones(2) */
  Eigen::VectorXd Configuration(uint32_t i) const {
    assert(i >= 0 && i <= (T_ + 1));
    return x_.segment(n_ * i, n_);
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

 protected:
  uint32_t n_;
  uint32_t T_;
  Eigen::VectorXd x_;
};
}  // namespace bewego
