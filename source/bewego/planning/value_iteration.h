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

#include <Eigen/Core>

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
 */
class ValueIteration {
 public:
  ValueIteration()
      : theta_(1e-6),
        max_iterations_(1e+5),
        // max_value_(100000),
        max_value_(std::numeric_limits<double>::max()),
        gamma_(1.) {}

  Eigen::MatrixXi solve(const Eigen::Vector2i& init,
                        const Eigen::Vector2i& goal,
                        const Eigen::MatrixXd& costmap) const;

  // returns value
  Eigen::MatrixXd Run(const Eigen::MatrixXd& costmap,
                      const Eigen::Vector2i& goal) const;

  void set_theta(double v) { theta_ = v; }
  void set_max_iterations(double v) { max_iterations_ = v; }

 private:
  double theta_;
  double max_iterations_;
  double max_value_;
  double gamma_;  // discount factor
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