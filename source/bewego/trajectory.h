// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#pragma once

#include <differentiable_map.h>

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
        for( auto f : functions_ )
            value += f->Forward(x);
        return value
    }

    void AddFunction(DifferentiableMapPtr f) {
        functions_.push_back(f)
    }

protected:
    std::vector<DifferentiableMapPtr> functions_;
}

/*!\brief Base class to implement a function network
        It allows to register functions and evaluates
        f(x_{i-1}, x_i, x_{i+1}) = \sum_i f_i(x_{i-1}, x_i, x_{i+1})
 */
class CliquesFunctionNetwork : public FunctionNetwork {
 
 public:

    CliquesFunctionNetwork(
        uint32_t input_dimension,
        uint32_t clique_element_dim) : 
        FunctionNetwork(),
        input_size_(input_dimension),
        nb_clique_elements_(3),
        clique_element_dim_(clique_element_dim),
        clique_dim_(nb_clique_elements_ * clique_element_dim_),
        nb_cliques_(uint32_t(input_size_ / clique_element_dim - 2))
        {
            functions_.resize(nb_cliques_);
            for( i=0; i<nb_cliques_; i++) {
                functions_[i] = []
            }
        }

    virtual uint32_t input_dimension() const { return input_size_;}
    virtual uint32_t nb_cliques() const { return nb_cliques_; }

    /** We call over all subfunctions in each clique */
    virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
        Eigen::VectorXd value = Eigen::VectorXd::Zero(1);
        for t, x_t in enumerate(self.all_cliques(x)):
            # print("x_c[{}] : {}".format(t, x_t))
            for f in self._functions[t]:
                value += f.forward(x_t)
        return value
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

        J = np.matrix(np.zeros((
            self.output_dimension(),
            self.input_dimension())))
        for t, x_t in enumerate(self.all_cliques(x)):
            for f in self._functions[t]:
                assert f.output_dimension() == self.output_dimension()
                c_id = t * self._clique_element_dim
                J[0, c_id:c_id + self._clique_dim] += f.jacobian(x_t)
        return J
    }

    /**
    The hessian matrix is of dimension m x m
                m (rows) : input size
                m (cols) : input size
    **/
    virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& q) const {
        H = np.matrix(np.zeros((
            self.input_dimension(),
            self.input_dimension())))
        dim = self._clique_dim
        for t, x_t in enumerate(self.all_cliques(x)):
            c_id = t * self._clique_element_dim
            for f in self._functions[t]:
                H[c_id:c_id + dim, c_id:c_id + dim] += f.hessian(x_t)
        return H
    }

     /**
        return the clique value
        TODO create a test using this function.
     **/
    virtual Eigen::VectorXd CliqueValue(t, x_t) {
        value = 0.
        for f in self._functions[t]:
            value += f.forward(x_t)
        return value
    }

    /**
        return the clique jacobian
        J : the full jacobian
     **/
    virtual Eigen::MatrixXd CliqueJacobian(J, t) {
        c_id = t * self._clique_element_dim
        return J[0, c_id:c_id + self._clique_dim]
    }

    /**
        return the clique hessian
        H : the full hessian
     **/
    virtual Eigen::MatrixXd CliqueHessian(H, t) {
        dim = self._clique_dim
        c_id = t * self._clique_element_dim
        return H[c_id:c_id + dim, c_id:c_id + dim]
    }

    // returns a list of all cliques
    std::vector<Eigen::VectorXd> AllCliques(const Eigen::VectorXd& x) {
        n = self._clique_element_dim
        dim = self._clique_dim
        clique_begin_ids = list(range(0, n * self._nb_cliques, n))
        cliques = [x[c_id:c_id + dim] for c_id in clique_begin_ids]
        assert len(cliques) == self._nb_cliques
        return cliques
    }

    // Register function f for clique i
    void RegisterFunctionForClique(t, f) {
        assert f.input_dimension() == self._clique_dim
        self._functions[t].append(f)
    }

    // Register function f
    void RegisterFunctionForAllAliques(f) {
        assert f.input_dimension() == self._clique_dim
        for t in range(self._nb_cliques):
            self._functions[t].append(f)
    }

    // Register function f
    void RegisterFunctionLastClique(f) {
        assert f.input_dimension() == self._clique_dim
        T = self._nb_cliques - 1
        self._functions[T].append(f)
    }

    // x_{t}
    DifferentiableMapPtr CenterOfCliqueMap() {
        dim = self._clique_element_dim
        return std::make_shared<RangeSubspaceMap>(
            dim * self._nb_clique_elements,
            list(range(dim, (self._nb_clique_elements - 1) * dim)));
    }

    // x_{t+1}
    DifferentiableMapPtr right_most_of_clique_map() {
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range((self._nb_clique_elements - 1) * dim,
                       self._nb_clique_elements * dim)));
    }

    // x_{t} ; x_{t+1}
    DifferentiableMapPtr right_of_clique_map() {
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range(dim, self._nb_clique_elements * dim)))
    }

    // x_{t-1}
    DifferentiableMapPtr left_most_of_clique_map() {
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range(0, dim)));
    }

    // x_{t-1} ; x_{t}
    DifferentiableMapPtr LeftOfCliqueMap() {
        dim = self._clique_element_dim
        return RangeSubspaceMap(
            dim * self._nb_clique_elements,
            list(range(0, (self._nb_clique_elements - 1) * dim)));
    }

private:

    uint32_t input_size_;
    uint32_t nb_clique_elements_;
    uint32_t clique_element_dim_;
    uint32_t clique_dim_ ;
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
