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
 *                                                             Thu 1 Apr 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/differentiable_map.h>

namespace bewego {

/**
 All DifferentiableMaps that integrate multiple sub-differentiable maps, should
 be declared as combination operators.

 This is to generate the combutational graph more
 efficiently Not declaring these maps Combination operators will not change the
 computational output but will definitly hinder performance when called on an
 optimization routine.
    */
class CombinationOperator : public DifferentiableMap {
 public:
  CombinationOperator() { is_atomic_ = false; }
  virtual VectorOfMaps nested_operators() const {
    throw std::runtime_error(
        "CombinationOperator does not implement nested_operators");
  }
  virtual VectorOfMaps ouput_operators() const { return VectorOfMaps(); }
  virtual VectorOfMaps input_operators() const { return nested_operators(); }
};

/** f round g : f(g(q))

    This function should be called pullback if we approxiate
    higher order (i.e., hessians) derivaties by pullback, here it's
    computing the true 1st order derivative of the composition.
*/
class Compose : public CombinationOperator {
 public:
  Compose(DifferentiableMapPtr f, DifferentiableMapPtr g) {
    // Make sure the composition makes sense
    if (g->output_dimension() != f->input_dimension()) {
      throw std::runtime_error("Compose : maps dimension missmatch");
    }
    f_ = f;
    g_ = g;
    type_ = "Compose";
  }

  uint32_t output_dimension() const { return f_->output_dimension(); }
  uint32_t input_dimension() const { return g_->input_dimension(); }

  virtual VectorOfMaps nested_operators() const {
    return VectorOfMaps({f_, g_});
  }
  virtual VectorOfMaps ouput_operators() const { return VectorOfMaps({f_}); }
  virtual VectorOfMaps input_operators() const { return VectorOfMaps({g_}); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
    return (*f_)((*g_)(q));
  }

  /** d/dq f(g(q)), applies chain rule.

            * J_f(g(q)) J_g

      If J is the jacobian of a function f(x), J_f = d/dx f(x)
        then the jacobian of the "pullback" of f defined on the
        range space of a map g, f(g(q)) is
                d/dq f(g(q)) = J_f(g(q)) J_g
        This method computes and
        returns this "pullback gradient" J_f (g(q)) J_g(q).

        WARNING: J_f is assumed to be a jacobian np.matrix object
    */
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
    return Evaluate(q).second;
  }

  /** d^2/dq^2 f(g(q)), applies chain rule.

            * J_g' H_f J_g + H_g J_f,

    https://en.wikipedia.org/wiki/Chain_rule (Higher derivatives)

    WARNING: If n > 1, where g : R^m -> R^n, we approximate the hessian
             to the first term. This is equivalent to considering H_g = 0
             It can be seen as operating a pullback of the curvature
             tensor of f by g.

    https://en.wikipedia.org/wiki/Pullback_(differential_geometry)
    */
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& q) {
    assert(f_->output_dimension() == 1);
    auto x = (*g_)(q);
    auto J_g = g_->Jacobian(q);
    Eigen::MatrixXd H = J_g.transpose() * f_->Hessian(x) * J_g;
    if (g_->output_dimension() == 1) {
      H += f_->Jacobian(x) * Eigen::VectorXd::Ones(input_dimension()) *
           g_->Hessian(q);
    }
    return H;
  }

  // d/dq f(g(q)), applies chain rule.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(
      const Eigen::VectorXd& q) const {
    auto g = g_->Evaluate(q);
    auto f_o_g = f_->Evaluate(g.first);
    return std::make_pair(f_o_g.first, f_o_g.second * g.second);
  }

 protected:
  std::shared_ptr<const DifferentiableMap> f_;
  std::shared_ptr<const DifferentiableMap> g_;
};

inline DifferentiableMapPtr ComposedWith(DifferentiableMapPtr f,
                                         DifferentiableMapPtr g) {
  return std::make_shared<Compose>(f, g);
}

class Scale : public CombinationOperator {
 public:
  Scale(DifferentiableMapPtr f, double alpha) : f_(f), alpha_(alpha) {
    type_ = "Scale";
  }

  uint32_t output_dimension() const { return f_->output_dimension(); }
  uint32_t input_dimension() const { return f_->input_dimension(); }

  virtual VectorOfMaps nested_operators() const { return VectorOfMaps({f_}); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return alpha_ * f_->Forward(x);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return alpha_ * f_->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    return alpha_ * f_->Hessian(x);
  }

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    if (other.type() != type_) {
      return false;
    } else {
      auto f = static_cast<const Scale&>(other);
      if (abs(f.alpha_ - alpha_) > 1e-6) return false;
      return true;
    }
  }

 protected:
  DifferentiableMapPtr f_;
  double alpha_;
};

inline DifferentiableMapPtr operator*(double scalar, DifferentiableMapPtr f) {
  return std::make_shared<Scale>(f, scalar);
}

class Offset : public CombinationOperator {
 public:
  Offset(DifferentiableMapPtr f, const Eigen::VectorXd& offset)
      : f_(f), offset_(offset) {
    assert(offset_.size() == f_->output_dimension());
    type_ = "Offset";
  }

  uint32_t output_dimension() const { return f_->output_dimension(); }
  uint32_t input_dimension() const { return f_->input_dimension(); }

  virtual VectorOfMaps nested_operators() const { return VectorOfMaps({f_}); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return f_->Forward(x) + offset_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return f_->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    return f_->Hessian(x);
  }

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    if (other.type() != type_) {
      return false;
    } else {
      auto f = static_cast<const Offset&>(other);
      return (f.offset_ - offset_).cwiseAbs().maxCoeff() < 1e-6;
    }
  }

 protected:
  DifferentiableMapPtr f_;
  Eigen::VectorXd offset_;
};

inline DifferentiableMapPtr operator+(DifferentiableMapPtr f,
                                      const Eigen::VectorXd& offset) {
  return std::make_shared<Offset>(f, offset);
}
inline DifferentiableMapPtr operator-(DifferentiableMapPtr f,
                                      const Eigen::VectorXd& offset) {
  return std::make_shared<Offset>(f, -offset);
}
inline DifferentiableMapPtr operator+(DifferentiableMapPtr f, double offset) {
  assert(f->output_dimension() == 1);
  return std::make_shared<Offset>(f, Eigen::VectorXd::Constant(1, offset));
}
inline DifferentiableMapPtr operator-(DifferentiableMapPtr f, double offset) {
  assert(f->output_dimension() == 1);
  return std::make_shared<Offset>(f, Eigen::VectorXd::Constant(1, -offset));
}

/**
 * \brief Represents the sum of a set of maps f_i.
 *
 * Details:
 *
 *   y(x) = \sum_{i=1}^N f_i(x)
 */
class SumMap : public CombinationOperator {
 public:
  SumMap() {
    maps_ = std::make_shared<VectorOfMaps>();
    type_ = "SumMap";
  }
  SumMap(std::shared_ptr<const VectorOfMaps> maps) : maps_(maps) {
    assert(maps_->size() > 0);
    for (uint32_t i = 0; i < maps_->size(); i++) {
      assert(maps_->at(i)->input_dimension() == input_dimension());
      assert(maps_->at(i)->output_dimension() == output_dimension());
    }
    PreAllocate();
    type_ = "SumMap";
  }

  virtual uint32_t input_dimension() const {
    return maps_->back()->input_dimension();
  }
  virtual uint32_t output_dimension() const {
    return maps_->back()->output_dimension();
  }

  virtual VectorOfMaps nested_operators() const { return *maps_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    y_.setZero();
    for (uint32_t i = 0; i < maps_->size(); i++) {
      y_ += maps_->at(i)->Forward(x);
    }
    return y_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    J_.setZero();
    for (uint32_t i = 0; i < maps_->size(); i++) {
      J_ += maps_->at(i)->Jacobian(x);
    }
    return J_;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    H_.setZero();
    for (uint32_t i = 0; i < maps_->size(); i++) {
      H_ += maps_->at(i)->Hessian(x);
    }
    return H_;
  }

  const VectorOfMaps& terms() const { return (*maps_); }

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    if (other.type() != type_) {
      return false;
    } else {
      auto f = static_cast<const SumMap&>(other);
      return f.maps_->size() == maps_->size();
    }
  }

 protected:
  std::shared_ptr<const VectorOfMaps> maps_;
};

inline DifferentiableMapPtr operator+(DifferentiableMapPtr f,
                                      DifferentiableMapPtr g) {
  auto maps = std::make_shared<VectorOfMaps>();
  maps->push_back(f);
  maps->push_back(g);
  return std::make_shared<SumMap>(maps);
}

inline DifferentiableMapPtr operator-(DifferentiableMapPtr f,
                                      DifferentiableMapPtr g) {
  auto maps = std::make_shared<VectorOfMaps>();
  maps->push_back(f);
  maps->push_back(-1. * g);
  return std::make_shared<SumMap>(maps);
}

/**
 * \brief Represents the sum of a set of maps f_i.
 *
 * Details:
 *
 *   f(x) = g(x) h(x)
 */
class ProductMap : public CombinationOperator {
 public:
  ProductMap(DifferentiableMapPtr f1, DifferentiableMapPtr f2)
      : g_(f1), h_(f2) {
    assert(f1->input_dimension() == f2->input_dimension());
    assert(f1->output_dimension() == 1);
    assert(f2->output_dimension() == 1);
    type_ = "ProductMap";
  }

  virtual uint32_t input_dimension() const { return g_->input_dimension(); }
  virtual uint32_t output_dimension() const { return 1; }

  virtual VectorOfMaps nested_operators() const {
    return VectorOfMaps({g_, h_});
  }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return (*g_)(x) * (*h_)(x);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    double v1 = (*g_)(x)[0];
    double v2 = (*h_)(x)[0];
    return v1 * h_->Jacobian(x) + v2 * g_->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(x.size() == input_dimension());
    double v1 = (*g_)(x)[0];
    double v2 = (*h_)(x)[0];
    Eigen::MatrixXd J1 = g_->Jacobian(x);
    Eigen::MatrixXd J2 = h_->Jacobian(x);
    Eigen::MatrixXd H = v1 * h_->Hessian(x) + v2 * g_->Hessian(x);
    return H + J1.transpose() * J2 + J2.transpose() * J1;
  }

 protected:
  DifferentiableMapPtr g_;
  DifferentiableMapPtr h_;
};

inline DifferentiableMapPtr operator*(DifferentiableMapPtr f,
                                      DifferentiableMapPtr g) {
  return std::make_shared<ProductMap>(f, g);
}

// Represent a function as f(x) = argmin_i g_i(x).
// All functions g_i must be of the same input dimensionality,
// as specified during construction.
// WARNING: This operator may lead to discontunious derivatives
class Min : public CombinationOperator {
 public:
  // All terms must be of dimension term_dimension.
  Min(uint32_t term_dimension) : term_dimension_(term_dimension) {
    type_ = "Min";
  }
  Min(const VectorOfMaps& v) {
    AddTerms(v);
    type_ = "Min";
  }
  virtual ~Min() {}

  virtual uint32_t input_dimension() const { return term_dimension_; }
  virtual uint32_t output_dimension() const { return 1; }

  virtual VectorOfMaps nested_operators() const { return functions_; }
  const VectorOfMaps& maps() const { return functions_; }

  void AddTerms(const VectorOfMaps& v) {
    assert(v.empty() != true);
    term_dimension_ = v.front()->input_dimension();
    for (auto& f : v) {
      assert(f->input_dimension() == term_dimension_);
      assert(f->output_dimension() == 1);
    }
    functions_ = v;
  }

  uint32_t GetMinFunctionId(const Eigen::VectorXd& x) const {
    double min = std::numeric_limits<double>::max();
    uint32_t min_id = 0;
    for (uint32_t i = 0; i < functions_.size(); i++) {
      double value = (*functions_[i])(x)[0];
      if (min > value) {
        min = value;
        min_id = i;
      }
    }
    return min_id;
  }

  // Evaluates f(x) = argmin_x (x).
  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    double min = std::numeric_limits<double>::max();
    for (auto& f : functions_) {
      double value = (*f)(x)[0];
      if (min > value) {
        min = value;
      }
    }
    return Eigen::VectorXd::Constant(1, min);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return functions_[GetMinFunctionId(x)]->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    return functions_[GetMinFunctionId(x)]->Hessian(x);
  }

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    if (other.type() != type_) {
      return false;
    } else {
      auto f = static_cast<const Min&>(other);
      bool eq_size = f.functions_.size() == functions_.size();
      bool eq_dim = f.term_dimension_ == term_dimension_;
      return eq_size && eq_dim;
    }
  }

 protected:
  VectorOfMaps functions_;
  uint32_t term_dimension_;
};

/** A smooth version of the distance function.
 *
 * Details:
 *
 *   f(d; \alpha) = sqrt(sq_dist + alpha^2) - alpha
 *
 *   since equality constraints are squared, using squared
 *   norms makes the optimization unstable. The regular norm
 *   is not smooth. Introduced by Tassa et al 2012 (IROS)
 *
 *   Takes as input the squared distance.
 */
class SoftDist : public CombinationOperator {
 public:
  SoftDist(DifferentiableMapPtr sq_dist, double alpha = .05);

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return sq_dist_->input_dimension(); }

  virtual VectorOfMaps nested_operators() const {
    return VectorOfMaps({sq_dist_});
  }

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    if (other.type() != type_) {
      return false;
    } else {
      auto f = static_cast<const SoftDist&>(other);
      if (abs(f.alpha_ - alpha_) > 1e-6) return false;
      if (abs(f.alpha_sq_ - alpha_sq_) > 1e-6) return false;
      return true;
    }
  }

 protected:
  DifferentiableMapPtr sq_dist_;
  double alpha_;
  double alpha_sq_;
};

/*! \brief Creates a combination of the maps
 *
 * Details:
 *
 *   phi(x) = [phi1(x); phi2(x); ...; phiN(x)]
 *
 * simply ``stacks" the maps output.
 * The hessian is not defined as this has > 1 output dimension
 */
class CombinedOutputMap : public CombinationOperator {
 public:
  CombinedOutputMap(const VectorOfMaps& maps) : maps_(maps), m_(0) {
    if (maps.empty()) {
      throw std::runtime_error("CombinedOutputMap : maps is empty");
    } else {
      uint32_t n = maps_.front()->input_dimension();
      for (auto m : maps) {
        m_ += m->output_dimension();
        assert(n == m->input_dimension());
      }
      PreAllocate();
      type_ = "CombinedOutputMap";
    }
  }

  uint32_t output_dimension() const { return m_; }
  uint32_t input_dimension() const { return maps_.front()->input_dimension(); }

  virtual VectorOfMaps nested_operators() const { return maps_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
    CheckInputDimension(q);
    uint32_t idx = 0;
    for (auto m : maps_) {
      y_.segment(idx, m->output_dimension()) = (*m)(q);
      idx += m->output_dimension();
    }
    return y_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
    CheckInputDimension(q);
    uint32_t idx = 0;
    for (auto map : maps_) {
      uint32_t m_map = map->output_dimension();
      uint32_t n_map = map->input_dimension();
      J_.block(idx, 0, m_map, n_map) = map->Jacobian(q);
      idx += m_map;
    }
    return J_;
  }

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    if (other.type() != type_) {
      return false;
    } else {
      auto f = static_cast<const CombinedOutputMap&>(other);
      bool eq_m = f.m_ == m_;
      bool eq_size = f.maps_.size() == maps_.size();
      return eq_m && eq_size;
    }
  }

 protected:
  uint32_t m_;
  VectorOfMaps maps_;
};

/*! \brief Evaluates multiple points map
 *
 * Details:
 *
 *   phi([x1; x2; ...; xN]) = [phi(x1); phi(x2); ...; phi(xN)]
 *
 * simply ``stacks" the maps output.
 * The hessian is not defined as this has > 1 output dimension
 */
class MultiEvalMap : public CombinationOperator {
 public:
  MultiEvalMap(DifferentiableMapPtr phi, uint32_t N) : phi_(phi), N_(N) {
    PreAllocate();
    type_ = "MultiEvalMap";
  }

  uint32_t output_dimension() const { return N_ * phi_->output_dimension(); }
  uint32_t input_dimension() const { return N_ * phi_->input_dimension(); }

  virtual VectorOfMaps nested_operators() const { return VectorOfMaps({phi_}); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
    CheckInputDimension(q);
    uint32_t m_map = phi_->output_dimension();
    uint32_t n_map = phi_->input_dimension();
    for (uint32_t i = 0; i < N_; i++) {
      y_.segment(i * m_map, m_map) = (*phi_)(q.segment(i * n_map, n_map));
    }
    return y_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
    CheckInputDimension(q);
    uint32_t m_map = phi_->output_dimension();
    uint32_t n_map = phi_->input_dimension();
    uint32_t idx = 0;
    for (uint32_t i = 0; i < N_; i++) {
      J_.block(i * m_map, i * n_map, m_map, n_map) =
          phi_->Jacobian(q.segment(i * n_map, n_map));
    }
    return J_;
  }

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    if (other.type() != type_) {
      return false;
    } else {
      auto f = static_cast<const MultiEvalMap&>(other);
      bool eq_m = f.N_ == N_;
      bool eq_phi = f.phi_->type() == phi_->type();
      return eq_phi && eq_phi;
    }
  }

 protected:
  uint32_t N_;
  DifferentiableMapPtr phi_;
};

/** The log barrier slice

    TODO make a combination operator
    */
class LogBarrierWithApprox : public LogBarrier {
 public:
  LogBarrierWithApprox(double max_hessian, double scalar = 1.)
      : max_hessian_(max_hessian) {
    SetScalar(scalar);
    type_ = "LogBarrierWithApprox";
  }

  void SetScalar(double scalar) {
    scalar_ = scalar;
    x_splice_ = sqrt(scalar_ / max_hessian_);
    approximation_ = MakeTaylorLogBarrier();
  }

  // Fix the hessian and gradient to constants
  std::shared_ptr<DifferentiableMap> MakeTaylorLogBarrier() const;

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

 protected:
  double max_hessian_;
  double scalar_;
  double x_splice_;
  std::shared_ptr<DifferentiableMap> approximation_;
};

inline DifferentiableMapPtr LogisticActivation(DifferentiableMapPtr f,
                                               double temp = 1) {
  return ComposedWith(std::make_shared<Logistic>(temp, 0., 1.), f);
}

/**
 * Computes the dot product between two maps
 *
 * Details:
 *
 *        f(q) = g_1(q)^T g_2(q)
 */
class DotProduct : public CombinationOperator {
 public:
  DotProduct(DifferentiableMapPtr map1, DifferentiableMapPtr map2);
  virtual ~DotProduct() {}

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  uint32_t input_dimension() const { return n_; }
  uint32_t output_dimension() const { return 1; }

  VectorOfMaps nested_operators() const { return VectorOfMaps({map1_, map2_}); }

 protected:
  DifferentiableMapPtr map1_;
  DifferentiableMapPtr map2_;
  uint32_t n_;
};

/*! \brief Implements an interpolation between functions
 *
 * Details:
 *
 *        f(x; e) = \sum_i e_i(x) * f_i(x)
 *
 * Note: this does not seem like a good name for this operator...
 * used to be called LinearCombine
 */
class ActivationWeights : public CombinationOperator {
 public:
  ActivationWeights(const VectorOfMaps& e, const VectorOfMaps& f);
  virtual ~ActivationWeights();

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  uint32_t input_dimension() const { return n_; }
  uint32_t output_dimension() const { return 1; }

  // TODO
  // VectorOfMaps nested_operators() const {
  // return VectorOfMaps({map1_, map2_}); }

 protected:
  uint32_t n_;
  VectorOfMaps e_times_f_;
};

/**
 * \brief Transitions smoothly between two functions
 *
 * Details:
 *
 *   sigma(x) = 1 /( 1 + exp(-k * phi(x) )
 *   f(x) = sigma(x) * f_1(x) + (1-sigma(x)) * f_2(x)
 *
 * The transition should occur at phi(x) = 0
 * Implements all of the variants of Evaluate() explicitly for efficiency.
 * https://en.wikipedia.org/wiki/Logistic_function
 *
 * Note: For k see TemperatureParameter bellow.
 */
class SmoothTransition : public ActivationWeights {
 public:
  SmoothTransition(DifferentiableMapPtr f1, DifferentiableMapPtr f2,
                   DifferentiableMapPtr phi, double k = 1)
      : ActivationWeights(
            std::vector<DifferentiableMapPtr>{
                LogisticActivation(phi, k),
                -1 * LogisticActivation(phi, k) + 1},
            std::vector<DifferentiableMapPtr>{f1, f2}),
        f1_(f1),
        f2_(f2),
        k_(k) {
    type_ = "SmoothTransition";
  }
  virtual ~SmoothTransition();

  /**
   * @brief TemperatureParameter
   * @param d : the size of the transition interval
   * @param threshold : how much mass is set on the function
   *                    t \in ]0, 1[
   * @return k : temperature parameter
   *
   * The temperature parameter or slope steepness, determines
   * how fast the transition between the two functions happens.
   * Here this parameter is set by giving an interval
   * in which to transition a certain percentage
   * of the function mass from f1 to f2.
   * By default we set that mass to 97% which means simply
   * a vast majority of the function mass.
   *
   * The parameter k is obtained as follows:
   *
   *                   t = 1 / [ 1 + exp(-kd/2)]
   *   t[1 + exp(-kd/2)] = 1
   *          exp(-kd/2) = (1 - t) / t
   *              -kd/2  = ln[(1 - t) / t]
   *                   k = -(2/d) ln[(1 - t) / t]
   */
  static double TemperatureParameter(double d, double threshold = .97) {
    return -(2 / d) * std::log(1 / threshold - 1);
  }

  DifferentiableMapPtr f1() const { return f1_; }
  DifferentiableMapPtr f2() const { return f2_; }
  double k() const { return k_; }

  // TODO
  // VectorOfMaps nested_operators() const {
  // return VectorOfMaps({map1_, map2_}); }

 protected:
  DifferentiableMapPtr f1_;
  DifferentiableMapPtr f2_;
  // Termperature parameter
  double k_;
};

};  // namespace bewego