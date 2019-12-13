// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <bewego/differentiable_map.h>

namespace bewego {

/** Simple identity map : f(x)=x **/
class IdentityMap : public DifferentiableMap {
public:
    IdentityMap(uint32_t n) : dim_(n) {}

    uint32_t output_dimension() const { return dim_; }
    uint32_t input_dimension() const { return dim_; }

    Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
        assert(q.size() == dim_);
        return q;
    }

    Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
        assert(q.size() == dim_);
        return Eigen::MatrixXd::Identity(dim_, dim_);
    }

    Eigen::MatrixXd Hessian(const Eigen::VectorXd& q) const {
        assert(q.size() == dim_);
        return Eigen::MatrixXd::Zero(dim_, dim_);
    }

protected:
    uint32_t dim_;
};

/** Test function that can be evaluated on a grid **/
class ExpTestFunction : public DifferentiableMap {
public:
    ExpTestFunction() {}

    uint32_t output_dimension() const { return 1; }
    uint32_t input_dimension() const { return 2; }

    Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
        assert(q.size() == 2);
        Eigen::VectorXd v(1);
        v << exp( -pow(2.0 * q[0], 2) - pow(0.5 * q[1], 2) );
        return v;
    }
};

}
