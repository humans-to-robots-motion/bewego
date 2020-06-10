// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include "value_iteration.h"
#include <iostream>

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

void ValueEightConnected(
    Eigen::VectorXd& neighbor_V,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& cost,
     uint32_t i, uint32_t j) {
  neighbor_V[0] = cost(i + 1, j) + V(i + 1, j);
  neighbor_V[1] = cost(i - 1, j) + V(i - 1, j);
  neighbor_V[2] = cost(i, j + 1) + V(i, j + 1);
  neighbor_V[3] = cost(i, j - 1) + V(i, j - 1);
  neighbor_V[4] = cost(i + 1, j + 1) * SQRT2 + V(i + 1, j + 1);
  neighbor_V[5] = cost(i - 1, j - 1) * SQRT2 + V(i - 1, j - 1);
  neighbor_V[6] = cost(i - 1, j + 1) * SQRT2 + V(i - 1, j + 1);
  neighbor_V[7] = cost(i + 1, j - 1) * SQRT2 + V(i + 1, j - 1);
}

Eigen::MatrixXd ValueIteration::Run(const Eigen::MatrixXd& costmap) const
{
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
  }
  std::cout << " -- max difference : " << diff << std::endl;
  return V_t;
}

}

