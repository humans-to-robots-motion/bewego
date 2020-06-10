// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <Eigen/Core>

namespace bewego {

class ValueIteration {
public:
    ValueIteration() : 
        theta_(1e-6),
        max_iterations_(1e+2) {}

    Eigen::MatrixXi solve(
        const Eigen::Vector2i& init,
        const Eigen::Vector2i& goal,
        const Eigen::MatrixXd& costmap) const { return Eigen::MatrixXi(); }

    // returns value
    Eigen::MatrixXd Run(const Eigen::MatrixXd& costmap) const;

    void set_theta(double v) { theta_ = v; }
    void set_max_iterations(double v) { max_iterations_ = v; }

private:
    double theta_;
    double max_iterations_;

};

}