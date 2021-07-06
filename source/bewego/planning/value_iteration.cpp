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

double SoftUpdate(const Eigen::VectorXd &q, double alpha = 1) {
  double q_softmax = 0;
  double Z = (alpha * q).array().exp().sum();
  Z = std::max(1e-40, Z);  // make sure we can still devide by Z.
  for (uint32_t k = 0; k < q.size(); k++) {
    q_softmax += q[k] * exp(alpha * q[k]) / Z;
  }
  return q_softmax;
}

double SoftMax(const Eigen::VectorXd &x, double alpha = 1) {
  double Z = (alpha * x).array().exp().sum();
  return (1. / alpha) * std::log(Z);
}

double SoftMin(const Eigen::VectorXd &x, double alpha = 1) {
  return SoftMax(x, -1 * alpha);
}

Eigen::VectorXd ValidIndicies(const Eigen::VectorXd &x,
                              const std::vector<bool> &valid_indices) {
  std::vector<double> a;
  for (uint32_t k = 0; k < valid_indices.size(); k++) {
    if (valid_indices[k]) {
      a.push_back(x[k]);
    }
  }
  return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(a.data(), a.size());
}

#define SQRT2 1.4142135623730951

std::vector<int> X = {1, -1, 0, 0, 1, -1, -1, 1};
std::vector<int> Y = {0, 0, 1, -1, 1, -1, 1, -1};

bool SetValidAction(uint32_t k, int x, int y, int rows, int cols,
                    std::vector<bool> &valid_actions) {
  valid_actions[k] = x >= 0 && x < rows && y >= 0 && y < cols;
  return valid_actions[k];
}

bool ValueEightConnected(Eigen::VectorXd &Q_values,
                         std::vector<bool> &valid_actions,
                         const Eigen::MatrixXd &V, const Eigen::MatrixXd &cost,
                         double gamma, uint32_t i, uint32_t j) {
  bool contains_invalid = false;
  // N, E, S, W
  for (uint32_t k = 0; k < 4; k++) {
    int x = i + X[k];
    int y = j + Y[k];
    if (SetValidAction(k, x, y, cost.rows(), cost.cols(), valid_actions)) {
      Q_values[k] = cost(x, y) + gamma * V(x, y);
    } else {
      contains_invalid = true;
    }
  }
  for (uint32_t k = 4; k < 8; k++) {
    uint32_t x = i + X[k];
    uint32_t y = j + Y[k];
    if (SetValidAction(k, x, y, cost.rows(), cost.cols(), valid_actions)) {
      Q_values[k] = cost(x, y) * SQRT2 + gamma * V(x, y);
    } else {
      contains_invalid = true;
    }
  }
  return contains_invalid;
}

Eigen::MatrixXd ValueIteration::Run(const Eigen::MatrixXd &costmap,
                                    const Eigen::Vector2i &goal) const {
  uint32_t m = costmap.rows();
  uint32_t n = costmap.cols();
  Eigen::MatrixXd cost = costmap;

  Eigen::MatrixXd V_t = Eigen::MatrixXd::Zero(m, n);
  Eigen::MatrixXd V_0 = Eigen::MatrixXd::Zero(m, n);

  uint32_t Na = 8;
  bool debug = false;

  cout << "alpha : " << alpha_ << endl;

  std::vector<bool> valid_actions(Na);
  Eigen::VectorXd q_values(Na);
  double diff = 0;
  uint32_t k = 0;
  for (k = 0; k < max_iterations_; k++) {
    diff = 0;
    // for each state
    for (uint32_t i = 0; i < m; i++) {
      for (uint32_t j = 0; j < n; j++) {
        if (i == goal.x() && j == goal.y()) {
          V_t(i, j) = cost(i, j);
        } else {
          // - 1) Calculate value for all actions
          bool has_invalid = ValueEightConnected(q_values, valid_actions, V_0,
                                                 cost, gamma_, i, j);
          if (has_invalid) {
            q_values = ValidIndicies(q_values, valid_actions);
          }

          // - 2) Take the min over actions
          V_t(i, j) = with_softmin_ ? SoftUpdate(q_values, -1 * alpha_)
                                    : q_values.minCoeff();
          if (has_invalid) {
            q_values.resize(Na);
          }

          // - 3) Compute magnitude of update
          double update = std::fabs(V_t(i, j) - V_0(i, j));

          if (debug && i == 40 && j == 40) {
            cout << "q_values : " << q_values.transpose() << endl;
            cout << "V_t(i, j) : " << V_t(i, j) << endl;
            cout << "V_0(i, j) : " << V_0(i, j) << endl;
            cout << "update : " << update << endl;
          }
          diff = std::max(update, diff);
        }
      }
    }
    V_0 = V_t;
    if (diff < theta_) {
      break;
    }
  }
  cout << " -- iterations : " << k << endl;
  cout << " -- max difference : " << diff << endl;
  return V_0.block(1, 1, m - 2, n - 2);
}  // namespace bewego

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

// -----------------------------------------------------------------------------
// QTable implementation.
// -----------------------------------------------------------------------------

QTable::~QTable() {}

// -----------------------------------------------------------------------------
// SoftQIteration implementation.
// -----------------------------------------------------------------------------

std::shared_ptr<QTable> SoftQIteration::Run(const Eigen::MatrixXd &costmap,
                                            const Eigen::Vector2i &goal) const {
  auto Q_t = std::make_shared<QTable>(costmap.rows(), costmap.cols(), 8);

  // 1) Calculate softvalue iteration
  auto value_iteration = std::make_shared<ValueIteration>();
  value_iteration->set_with_softmin(true);

  // 2) Setup q-table

  return Q_t;
}

}  // namespace bewego
