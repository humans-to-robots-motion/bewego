// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

namespace bewego {

class ValueIteration {

    Eigen::MatrixXi solve(
        const Eigen::Vectorid& s,
        const Eigen::Vectorid& t,
        const Eigen::MatrixXd& costmap) const;

    void Run(const Eigen::MatrixXd& costmap) const;
    void set_theta(double v) { theta_ = v; }

};

}