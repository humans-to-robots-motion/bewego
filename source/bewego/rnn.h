// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Phliipp Kratzer, philipp.kratzer@ipvs.uni-stuttgart.de
// rnn.h: Implementations of Recurrent neural network structures and
// analytical input->output Jacobians for fast computations.

#pragma once

#include <bewego/differentiable_map.h>
#include <tuple>
#include <vector>
#include <memory>
#include "rotations.h"

namespace bewego {

  static double Sigmoid(const double x)  {
    if (x > 0)
      return 1./(1. + exp(-x));
    else
      return exp(x) / ( 1. + exp(x) );
  }


  /**
      Abstract class for a RNN cell. A RNN cell maps a pair (input, hidden state) to a pair (output, new hidden state)
  **/
  class RNNCell {
  public:
    virtual uint32_t output_dimension() const = 0;
    virtual uint32_t hidden_dimension() const = 0;
    virtual uint32_t input_dimension() const = 0;

    /** maps input in and hidden state h to output o and new hidden state h1**/
    virtual std::tuple<Eigen::VectorXd, Eigen::VectorXd> Forward(const Eigen::VectorXd& in, const Eigen::VectorXd& h) const = 0;
    /** outputs input->output Jacobians of the RNN: dodin dodh dhdin dhdh **/
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Jacobian(const Eigen::VectorXd& in, const Eigen::VectorXd& h) const = 0;
  };


  /**
      Abstract class for a coupled RNN cell (coupled output and hidden state). A coupled RNN cell maps a pair (input, hidden state) to an output
  **/
  class CoupledRNNCell {
  public:
    virtual uint32_t output_dimension() const = 0;
    virtual uint32_t hidden_dimension() const = 0;
    virtual uint32_t input_dimension() const = 0;

    /** maps input in and hidden state h to output o **/
    virtual Eigen::VectorXd Forward(const Eigen::VectorXd& in, const Eigen::VectorXd& h) const = 0;
    /** outputs input->output Jacobians of the coupled RNN: dodin dodh **/
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> Jacobian(const Eigen::VectorXd& in, const Eigen::VectorXd& h) const = 0;
  };


  /**
      Gated Recurrent Unit. A RNN cell with coupled output and hidden state.
  **/
  class GRUCell : public CoupledRNNCell {
  public:
    uint32_t output_dimension() const {return hidden_dim;};
    uint32_t hidden_dimension() const {return hidden_dim;};
    uint32_t input_dimension() const {return input_dim;};

    GRUCell(const Eigen::MatrixXd& Wi, const Eigen::MatrixXd& Wr, const Eigen::MatrixXd& Wn, const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rr, const Eigen::MatrixXd& Rn, const Eigen::VectorXd& bWi, const Eigen::VectorXd& bWr, const Eigen::VectorXd& bWn, const Eigen::VectorXd& bRi, const Eigen::VectorXd& bRr, const Eigen::VectorXd& bRn);

    Eigen::VectorXd Forward(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const;

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const;
  protected:
    uint32_t input_dim;
    uint32_t hidden_dim;
    Eigen::MatrixXd Wi;
    Eigen::MatrixXd Wr;
    Eigen::MatrixXd Wn;
    Eigen::MatrixXd Ri;
    Eigen::MatrixXd Rr;
    Eigen::MatrixXd Rn;
    Eigen::VectorXd bWi;
    Eigen::VectorXd bWr;
    Eigen::VectorXd bWn;
    Eigen::VectorXd bRi;
    Eigen::VectorXd bRr;
    Eigen::VectorXd bRn;

  };


  /**
      Class for stacking RNN cells. Stacks multiple coupled RNNs + a linear layer on top.
   **/
  class StackedCoupledRNNCell : public RNNCell {
  public:
    uint32_t output_dimension() const {return output_dim;}
    uint32_t hidden_dimension() const {return layers*hidden_dim;}
    uint32_t input_dimension() const {return cells[0]->input_dimension();}

    StackedCoupledRNNCell(const int layers, const int hidden_dim, const int output_dim, const std::vector<std::shared_ptr<CoupledRNNCell>>& cells, const Eigen::MatrixXd& Wdense, const Eigen::VectorXd& bdense);

    std::tuple<Eigen::VectorXd, Eigen::VectorXd> Forward(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const;

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const;

  protected:
    std::vector<std::shared_ptr<CoupledRNNCell>> cells;
    int layers;
    int hidden_dim;
    int output_dim;
    Eigen::MatrixXd Wdense;
    Eigen::VectorXd bdense;
  };


  /**
      Unrolled position-velocity RNN. Reimplemantation of a position-velocity encoder-decoder network for human movement prediction (https://arxiv.org/abs/1906.06514) with adaptions based on (https://arxiv.org/abs/1910.01843).
  **/
  class VRED {
  public:
    VRED(std::shared_ptr<RNNCell>& cell, int dim_trans);

    /** Returns a prediction for future timesteps given control parameters deltas **/
    Eigen::MatrixXd Forward (const Eigen::MatrixXd data, const Eigen::MatrixXd deltas, int src_length, int pred_length) const;

    /** Jacobian wrt. control parameters delta as matrix **/
    Eigen::MatrixXd Jacobian (const Eigen::MatrixXd data, const Eigen::MatrixXd deltas, int src_length, int pred_length) const;

  protected:
    int dim_trans;
    std::shared_ptr<DifferentiableMap> eulertoquat, quattoexpmap, expmaptoquat, quattoeuler;
    std::shared_ptr<RNNCell> cell;
  };

}
