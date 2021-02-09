// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <Eigen/Core>
#include <cassert>

namespace bewego {

class DifferentiableMap {
 public:
  virtual uint32_t output_dimension() const = 0;
  virtual uint32_t input_dimension() const = 0;

  /** Method called when call object */
  Eigen::VectorXd operator()(const Eigen::VectorXd& q) const {
    return Forward(q);
  }

  /** Should return an array or single value */
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& q) const = 0;

  /** Should return an array or single value
          n : input dimension
      Convienience function to get gradients
      in the same shape as the input vector
      for addition and substraction, of course gradients are
      only availables if the output dimension is one.
  */
  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& q) const {
    assert(output_dimension() == 1);
    return Jacobian(q).row(0);
  }

  /** Should return a matrix or single value of
              m x n : ouput x input (dimensions)
          by default the method returns the finite difference jacobian.
          WARNING the object returned by this function is a numpy matrix.
  */
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
    return FiniteDifferenceJacobian(*this, q);
  }

  /** Should return the hessian matrix
              n x n : input x input (dimensions)
          by default the method returns the finite difference hessian
          that relies on the jacobian function.
          This method would be a third order tensor
          in the case of multiple output, we exclude this case for now.
          WARNING the object returned by this function is a numpy matrix.
          */
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& q) const {
    return FiniteDifferenceHessian(*this, q);
  }

  /** Evaluates the map and jacobian simultaneously. The default
          implementation simply calls both forward and Getjacobian()
          separately but overriding this method can make the evaluation
          more efficient
          */
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(
      const Eigen::VectorXd& q) const {
    return std::make_pair(Forward(q), Jacobian(q));
  }

  /** Takes an object f that has a forward method returning
      a numpy array when querried.
      */
  static Eigen::MatrixXd FiniteDifferenceJacobian(const DifferentiableMap& f,
                                                  const Eigen::VectorXd& q);

  /** Takes an object f that has a forward method returning
      a numpy array when querried.
      */
  static Eigen::MatrixXd FiniteDifferenceHessian(const DifferentiableMap& f,
                                                 const Eigen::VectorXd& q);

  /** check against finite differences */
  bool CheckJacobian(double precision = 1e-12) const;

  /** check against finite differences */
  bool CheckHessian(double precision = 1e-12) const;
};

using DifferentiableMapPtr = std::shared_ptr<const DifferentiableMap>;

}  // namespace bewego