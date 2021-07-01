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
 *                                                             Thu 11 Feb 2021
 */

// author: Jim Mainprice, mainprice@gmail.com

#include "value_iteration.h"

#include <iostream>
#include <limits>
#include <vector>

using std::cout;
using std::endl;

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
// 19:                       Vk[s] = max_a ∑_s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])
// 20:           until ∀s |Vk[s]-Vk-1[s]| < θ
// 21:           for each state s do
// 22:                 π[s] = argmaxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk[s'])
// 23:           return π,Vk

namespace bewego {

#define SQRT2 1.4142135623730951

std::vector<int> X = {1, -1, 0, 0, 1, -1, -1, 1};
std::vector<int> Y = {0, 0, 1, -1, 1, -1, 1, -1};

void ValueEightConnected(Eigen::VectorXd &neighbor_V, const Eigen::MatrixXd &V,
                         const Eigen::MatrixXd &cost, double gamma, uint32_t i,
                         uint32_t j) {
  // N, E, S, W
  for (uint32_t k = 0; k < 4; k++) {
    uint32_t x = i + X[k];
    uint32_t y = j + Y[k];
    neighbor_V[k] = cost(x, y) + gamma * V(x, y);
  }
  for (uint32_t k = 4; k < 8; k++) {
    uint32_t x = i + X[k];
    uint32_t y = j + Y[k];
    neighbor_V[k] = cost(x, y) * SQRT2 + gamma * V(x, y);
  }
}

Eigen::MatrixXd ValueIteration::Run(const Eigen::MatrixXd &costmap,
                                    const Eigen::Vector2i &goal) const {
  uint32_t m = costmap.rows() + 2;
  uint32_t n = costmap.cols() + 2;
  Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(m, n);
  cost.block(1, 1, m - 2, n - 2) = costmap;

  Eigen::MatrixXd V_t = Eigen::MatrixXd::Zero(m, n);
  Eigen::MatrixXd V_0 = Eigen::MatrixXd::Zero(m, n);

  // Pad with +inf cost
  V_0.row(0) = max_value_ * Eigen::VectorXd::Ones(n);
  V_0.row(m - 1) = max_value_ * Eigen::VectorXd::Ones(n);
  V_0.col(0) = max_value_ * Eigen::VectorXd::Ones(m);
  V_0.col(n - 1) = max_value_ * Eigen::VectorXd::Ones(m);

  V_t = V_0;

  Eigen::VectorXd neighbor_costs(8);
  double diff = 0;
  for (uint32_t k = 0; k < max_iterations_; k++) {
    diff = 0;
    // for each state
    for (uint32_t i = 1; i < m - 1; i++) {
      for (uint32_t j = 1; j < n - 1; j++) {
        if (i == goal.x() + 1 && j == goal.y() + 1) {
          V_t(i, j) = cost(i, j);
        } else {
          ValueEightConnected(neighbor_costs, V_0, cost, gamma_, i, j);
          V_t(i, j) = neighbor_costs.minCoeff();
          double update = std::fabs(V_t(i, j) - V_0(i, j));
          // if (i == 1 && j == 10) {
          //   cout << "neighbor_costs : " << neighbor_costs.transpose() <<
          //   endl; cout << "V_t(i, j) : " << V_t(i, j) << endl; cout <<
          //   "V_0(i, j) : " << V_0(i, j) << endl; cout << "update : " <<
          //   update << endl;
          // }
          diff = std::max(update, diff);
        }
      }
    }
    V_0 = V_t;
    if (diff < theta_) {
      std::cout << " -- iterations : " << k << std::endl;
      break;
    }
  }
  std::cout << " -- max difference : " << diff << std::endl;
  return V_0.block(1, 1, m - 2, n - 2);
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
  Eigen::MatrixXd costmap2 =
      Eigen::MatrixXd::Zero(costmap.rows(), costmap.cols());
  costmap2(goal.x(), goal.y()) = -1e+3;
  Eigen::MatrixXd V = Run(costmap2, goal);
  std::vector<Eigen::Vector2i> path = {init};
  while (path.back().x() != goal.x() && path.back().y() != goal.y()) {
    path.push_back(MinNeighbor(V, path.back()));
  }

  Eigen::MatrixXi path_m(path.size(), 2);  // Initialize path
  for (int k = 0; k < path.size(); k++) {
    path_m(k, 0) = path[k].x();
    path_m(k, 1) = path[k].y();
  }

  return path_m;
}

}  // namespace bewego
