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
/**
class TrajectoryObjectiveFunction(DifferentiableMap) {

    TrajectoryObjectiveFunction(q_init, function_network) {
        self._q_init = q_init
        self._n = q_init.size
        self._function_network = function_network
    }

    Eigen::VectorXd full_vector(self, x_active) {
        assert x_active.size == (
            self._function_network.input_dimension() - self._n)
        x_full = np.zeros(self._function_network.input_dimension())
        x_full[:self._n] = self._q_init
        x_full[self._n:] = x_active
        return x_full
    }

    uint32_t output_dimension(self) {
        return self._function_network.output_dimension()
    }

    uint32_t input_dimension(self) {
        return self._function_network.input_dimension() - self._n
    }

    Eigen::VectorXd Forward(self, x) {
        x_full = self.full_vector(x)
        return min(1e100, self._function_network(x_full))
    }

    Eigen::MatrixX Jacobian(self, x) {
        x_full = self.full_vector(x)
        return self._function_network.jacobian(x_full)[0, self._n:]
    }

    Eigen::MatrixX Hessian(self, x) {
        x_full = self.full_vector(x)
        H = self._function_network.hessian(x_full)[self._n:, self._n:]
        return np.array(H)
    }

protected:
    uint32_t q_init_;
    uint32_t n_;
    uint32_t function_network_;
};
*/

}  // namespace bewego
