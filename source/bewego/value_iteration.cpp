// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include "value_iteration.h"

#include <iostream>
#include <limits>
#include <vector>

// 1: Procedure Value_Iteration(S,A,P,R,θ)
// 2:           Inputs
// 3:                     S is the set of all states
// 4:                     A is the set of all actions
// 5:                     P is state transition function specifying P(s'|s,a)
// 6:                     R is a reward function R(s,a,s')
// 7:                     θ a threshold, θ>0
// 8:           Output
// 9:                     π[S] approximately optimal policy
// 10:                    V[S] value function
// 11:           Local
// 12:                     real array Vk[S] is a sequence of value functions
// 13:                     action array π[S]
// 14:           assign V0[S] arbitrarily
// 15:           k ←0
// 16:           repeat
// 17:                 k ←k+1
// 18:                 for each state s do
// 19:                       Vk[s] = maxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])
// 20:           until ∀s |Vk[s]-Vk-1[s]| < θ
// 21:           for each state s do
// 22:                 π[s] = argmaxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk[s'])
// 23:           return π,Vk

namespace bewego {

#define SQRT2 1.4142135623730951

std::vector<int> X = {1, -1, 0, 0, 1, -1, -1, 1};
std::vector<int> Y = {0, 0, 1, -1, 1, -1, 1, -1};

void ValueEightConnected(Eigen::VectorXd &neighbor_V, const Eigen::MatrixXd &V,
                         const Eigen::MatrixXd &cost, uint32_t i, uint32_t j) {
  neighbor_V[0] = V(i + X[0], j + Y[0]) + cost(i + X[0], j + Y[0]);
  neighbor_V[1] = V(i + X[1], j + Y[1]) + cost(i + X[1], j + Y[1]);
  neighbor_V[2] = V(i + X[2], j + Y[2]) + cost(i + X[2], j + Y[2]);
  neighbor_V[3] = V(i + X[3], j + Y[3]) + cost(i + X[3], j + Y[3]);
  neighbor_V[4] = V(i + X[4], j + Y[4]) + cost(i + X[4], j + Y[4]) * SQRT2;
  neighbor_V[5] = V(i + X[5], j + Y[5]) + cost(i + X[5], j + Y[5]) * SQRT2;
  neighbor_V[6] = V(i + X[6], j + Y[6]) + cost(i + X[6], j + Y[6]) * SQRT2;
  neighbor_V[7] = V(i + X[7], j + Y[7]) + cost(i + X[7], j + Y[7]) * SQRT2;
}

Eigen::MatrixXd ValueIteration::Run(const Eigen::MatrixXd &costmap) const {
  uint32_t m = costmap.rows();
  uint32_t n = costmap.cols();
  Eigen::MatrixXd V_t = Eigen::MatrixXd::Zero(m, n);
  Eigen::MatrixXd V_0 = Eigen::MatrixXd::Zero(m, n);
  Eigen::VectorXd neighbor_costs(8);
  double diff = 0;
  for (uint32_t k = 0; k < max_iterations_; k++) {
    diff = 0;
    for (uint32_t i = 1; i < m - 1; i++) {
      for (uint32_t j = 1; j < n - 1; j++) {
        ValueEightConnected(neighbor_costs, V_t, costmap, i, j);
        V_t(i, j) = neighbor_costs.minCoeff();
        diff = std::max(std::abs(V_t(i, j) - V_0(i, j)), diff);
      }
    }
    V_0 = V_t;
    if (diff < theta_) {
      break;
    }
  }
  std::cout << " -- max difference : " << diff << std::endl;
  return V_t;
}

Eigen::Vector2i MinNeighbor(const Eigen::MatrixXd &V,
                            const Eigen::Vector2i &coord) {
  uint32_t i = coord.x();
  uint32_t j = coord.y();
  uint32_t index;
  double min_v = std::numeric_limits<double>::max();
  for (uint32_t k = 0; k < 8; k++) {
    double value = V(i + X[k], j + Y[k]);
    if (value < min_v) {
      min_v = value;
      index = k;
    }
  }
  return Eigen::Vector2i(i + X[index], j + Y[index]);
}

Eigen::MatrixXi ValueIteration::solve(const Eigen::Vector2i &init,
                                      const Eigen::Vector2i &goal,
                                      const Eigen::MatrixXd &costmap) const {
  Eigen::MatrixXd V = Run(costmap);
  std::vector<Eigen::Vector2i> path = {init};
  while (path.back() == goal) {
    path.push_back(MinNeighbor(V, path.back()));
  }

  Eigen::MatrixXi path_m(path.size(), 2);  // Initialize path
  for (int k = 0; k < path.size(); k++) {
    path_m(k, 2) = path[k].x();
    path_m(k, 1) = path[k].y();
  }

  return path_m;
}

}  // namespace bewego
