/*
 * Copyright (c) 2021
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
 *                                                              Thu 15 Apr 2021
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/derivatives/differentiable_map.h>

#include <Eigen/Geometry>

namespace bewego {

//! Gets the position from state
class Position : public DifferentiableMap {
 public:
  Position(uint32_t n) : n_(n) {
    J_ = Eigen::MatrixXd::Zero(output_dimension(), input_dimension());
    J_.block(0, 0, n_, n_) = Eigen::MatrixXd::Identity(n_, n_);
  }
  virtual ~Position();
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual uint32_t input_dimension() const { return 2 * n_; }
  virtual uint32_t output_dimension() const { return n_; }

 protected:
  uint32_t n_;
};

//! Gets the velocity from state
class Velocity : public DifferentiableMap {
 public:
  Velocity(uint32_t n) : n_(n) {
    J_ = Eigen::MatrixXd::Zero(output_dimension(), input_dimension());
    J_.block(0, n_, n_, n_) = Eigen::MatrixXd::Identity(n_, n_);
  }
  virtual ~Velocity();
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual uint32_t input_dimension() const { return 2 * n_; }
  virtual uint32_t output_dimension() const { return n_; }

 protected:
  uint32_t n_;
};

/*
 * Computes the velocity in some frame given as input
 * The frame itself is subject to change, which is why it is part of the
 * function input. For now we constrain this function to take 24
 * numbers as input, which are the position and orientation of the end-effector
 * (12) with their derivatives x = (q, \dot q).
 *
 * We only use this function for computing the velocity
 * in the end-effector frame, but internaly it simply does
 *
 *     y = f([X1 x2]) = X1^T x2
 *
 * and thus we could have an abstract differentiable map of this form
 * defined somewhere else.
 */
class VelocityInFrame : public DifferentiableMap {
 public:
  VelocityInFrame() {
    J_ = Eigen::MatrixXd::Zero(output_dimension(), input_dimension());
  }
  virtual ~VelocityInFrame() {}

  /*!\brief Evaluates x = phi(q) and returns x in the return parameter.
   * Derived classes should validate that x isn't null.
   */
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;

  /*!\brief Evaluates the Jacobian of the map at q (J = d/dq phi(q)) and
   * returns the resulting matrix at J. Derived classes should validate that J
   * isn't null.
   */
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;

  virtual uint32_t input_dimension() const { return 24; }
  virtual uint32_t output_dimension() const { return 3; }
};

/*
 * Homeogeneous transformation as DifferentiableMap

        Takes an angle and rotates the point p0 by this angle

            f(q) = T(q) * p_0

        where T defines a rotation and translation (3DoFs)
            q_{0,1}     => translation
            q_{2}       => rotation

                T = [ R(q)  p(q) ]
                    [ 0 0    1   ]
 */
class HomogeneousTransform2d : public DifferentiableMap {
 public:
  HomogeneousTransform2d(const Eigen::Vector2d& p) : p0_(p) {
    J_ = Eigen::MatrixXd::Zero(output_dimension(), input_dimension());
  }
  virtual ~HomogeneousTransform2d() {}

  /*!\brief Evaluates x = phi(q) and returns x in the return parameter.
   * Derived classes should validate that x isn't null.
   */
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& q) const;
  /*!\brief Evaluates the Jacobian of the map at q (J = d/dq phi(q)) and
   * returns the resulting matrix at J. Derived classes should validate that J
   * isn't null.
   */
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const;

  virtual uint32_t input_dimension() const { return 3; }
  virtual uint32_t output_dimension() const { return 2; }

  Eigen::Vector2d p0_;
};

/*
 * Homeogeneous transformation as DifferentiableMap

        Takes an angle and rotates the point p0 by this angle

            f(q) = T(q) * p_0

        where T defines a rotation and translation (6DoFs)
            q_{0,1}     => translation
            q_{2}       => rotation

                T = [ R(q)  p(q) ]
                    [ 0 0    1   ]

   We use the Euler angel convention 3-2-1, which is found
   TODO it would be nice to match the ROS convention
   we simply use this one because it was available as derivation
   in termes for sin and cos. It seems that ROS dos
   Static-Z-Y-X so it should be the same. Still needs to test.
 */
class HomogeneousTransform3d : public DifferentiableMap {
 public:
  HomogeneousTransform3d(const Eigen::Vector3d& p) : p0_(p) {
    J_ = Eigen::MatrixXd::Zero(output_dimension(), input_dimension());
  }
  virtual ~HomogeneousTransform3d() {}

  /*!\brief Evaluates x = phi(q) and returns x in the return parameter.
   * Derived classes should validate that x isn't null.
   */
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& q) const;

  /*!\brief Evaluates the Jacobian of the map at q (J = d/dq phi(q)) and
   * returns the resulting matrix at J. Derived classes should validate that J
   * isn't null.
   */
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const;

  // Return Transform
  Eigen::Isometry3d Transform(const Eigen::VectorXd& q) const;

  virtual uint32_t input_dimension() const { return 6; }
  virtual uint32_t output_dimension() const { return 3; }

  Eigen::Vector3d p0_;
};

}  // namespace bewego