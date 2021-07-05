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

#pragma once

// author: Jim Mainprice, mainprice@gmail.com

#include <Eigen/Core>
#include <vector>

namespace bewego {

/**
 * \brief Value Iteration on costmaps
 *        The planning problem is assumed to be 8 connected
 *        We condier cost instead of reward, the aglorithm is is
 *        is sumarized bellow:
 *
 *    repeat
 *      k ←k+1
 *      for each state s do
 *          Vk[s] = min_a ∑_s' P(s'|s,a) (C(s,a,s')+ γVk-1[s'])
 *    until ∀s |Vk[s]-Vk-1[s]| < θ
 *
 *    Can operate softvalue iteration.
 */
class ValueIteration {
 public:
  ValueIteration()
      : theta_(1e-6),
        max_iterations_(1e+5),
        max_value_(std::numeric_limits<double>::max()),
        gamma_(1.),
        with_softmin_(false),
        alpha_(1.) {}

  Eigen::MatrixXi solve(const Eigen::Vector2i& init,
                        const Eigen::Vector2i& goal,
                        const Eigen::MatrixXd& costmap) const;

  // returns value
  Eigen::MatrixXd Run(const Eigen::MatrixXd& costmap,
                      const Eigen::Vector2i& goal) const;

  void set_theta(double v) { theta_ = v; }
  void set_max_iterations(double v) { max_iterations_ = v; }
  void set_with_softmin(bool v) { with_softmin_ = v; }
  void set_softmin_alpha(double v) { alpha_ = v; }

 private:
  double theta_;
  double max_iterations_;
  double max_value_;
  double gamma_;  // discount factor
  bool with_softmin_;
  double alpha_;  // softmin strength
};

/**
 * \brief Q function
 *
 * This class defines a generic descrete Q function for states that
 * can be stored in matrix form.
 */
class QTable {
 public:
  QTable(uint32_t rows, uint32_t cols, uint32_t nb_actions) {
    q_.resize(nb_actions);
    for (uint32_t k = 0; k < q_.size(); k++) {
      q_[k] = Eigen::MatrixXd::Zero(rows, cols);
    }
  }

  virtual ~QTable();

  double operator()(uint32_t i, uint32_t j, uint32_t a) const {
    return q_[a](i, j);
  }
  double& operator()(uint32_t i, uint32_t j, uint32_t a) { return q_[a](i, j); }

  /** return the index of the max action for that state **/
  uint32_t argmax(uint32_t i, uint32_t j) const {
    uint32_t max_idx = 0;
    double max = std::numeric_limits<double>::lowest();
    for (uint32_t k = 0; k < q_.size(); k++) {
      if (q_[k](i, j) > max) {
        max_idx = k;
      }
    }
    return max_idx;
  }

  /** Returns the max value at a certain axion **/
  double max(uint32_t i, uint32_t j) const { return q_[argmax(i, j)](i, j); }

  /** Return values for display **/
  std::vector<Eigen::MatrixXd> values() const { return q_; }

 private:
  std::vector<Eigen::MatrixXd> q_;
};

/**
 * \brief Soft Q Iteration on costmaps
 *        here the underlying assumtion is that we have a softmin policy
 */
class SoftQIteration {
 public:
  SoftQIteration()
      : theta_(1e-6),
        max_iterations_(1e+5),
        // max_value_(100000),
        max_value_(std::numeric_limits<double>::max()),
        gamma_(1.) {}

  // returns value
  std::shared_ptr<QTable> Run(const Eigen::MatrixXd& costmap,
                              const Eigen::Vector2i& goal) const;

 private:
  double theta_;
  double max_iterations_;
  double max_value_;
  double gamma_;  // discount factor
};

/**
 * \brief State Vistation Frequency on costmaps
 *        here the underlying assumtion is that we have a softmin policy
 */
class StateVistationFrequency {
 public:
  StateVistationFrequency() {}
};

}  // namespace bewego