#pragma once

#include <bewego/differentiable_map.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <memory>

using std::cout;
using std::endl;

namespace bewego {

  /** Abstract class for a RNN cell. A RNN cell maps a pair (input, hidden state) to a pair (output, new hidden state) **/
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


  /** Gated Recurrent Unit. A RNN cell with coupled output and hidden state. **/
  class GRUCell : public RNNCell {
  public:
    uint32_t output_dimension() const {return hidden_dim;};
    uint32_t hidden_dimension() const {return hidden_dim;};
    uint32_t input_dimension() const {return input_dim;};

    GRUCell(const int input_dimension, const int hidden_dimension, const Eigen::MatrixXd& Wi, const Eigen::MatrixXd& Wr, const Eigen::MatrixXd& Wn, const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rr, const Eigen::MatrixXd& Rn, const Eigen::VectorXd& bWi, const Eigen::VectorXd& bWr, const Eigen::VectorXd& bWn, const Eigen::VectorXd& bRi, const Eigen::VectorXd& bRr, const Eigen::VectorXd& bRn) {
      this->input_dim = input_dimension;
      this->hidden_dim = hidden_dimension;
      this->Wi = Wi;
      this->Wr = Wr;
      this->Wn = Wn;
      this->Ri = Ri;
      this->Rr = Rr;
      this->Rn = Rn;
      this->bWi = bWi;
      this->bWr = bWr;
      this->bWn = bWn;
      this->bRi = bRi;
      this->bRr = bRr;
      this->bRn = bRn;
    }

    static double Sigmoid(const double x)  {
      if (x > 0)
	return 1./(1. + exp(-x));
      else
	return exp(x) / ( 1. + exp(x) );
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXd> Forward(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {
      Eigen::VectorXd r = (Wr * x + Rr * h_in + bWr + bRr).unaryExpr(&Sigmoid);
      Eigen::VectorXd i = (Wi * x + Ri * h_in + bWi + bRi).unaryExpr(&Sigmoid);
      Eigen::VectorXd n = (Wn * x + r.cwiseProduct(Rn * h_in + bRn) + bWn).unaryExpr(&tanh);
      Eigen::VectorXd h_out = (Eigen::VectorXd::Ones(hidden_dim) - i).cwiseProduct(n) + i.cwiseProduct(h_in);
      return {h_out, h_out};
    }

    /* because output and h are coupled, only 2 matrices are returned */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {
      Eigen::VectorXd r = (Wr * x + Rr * h_in + bWr + bRr).unaryExpr(&Sigmoid);
      Eigen::VectorXd dsdr = r.cwiseProduct(Eigen::VectorXd::Ones(hidden_dim)-r);
      Eigen::MatrixXd drdx;
      drdx.array() = Wr.array().colwise() * dsdr.array();
      Eigen::MatrixXd drdh;
      drdh.array() = Rr.array().colwise() * dsdr.array();

      Eigen::VectorXd i = (Wi * x + Ri * h_in + bWi + bRi).unaryExpr(&Sigmoid);
      Eigen::VectorXd dsdi = i.cwiseProduct(Eigen::VectorXd::Ones(hidden_dim)-i);
      Eigen::MatrixXd didx;

      didx.array() = Wi.array().colwise() * dsdi.array();
      Eigen::MatrixXd didh;
      didh.array() = Ri.array().colwise() * dsdi.array();

      Eigen::VectorXd n = (Wn * x + r.cwiseProduct(Rn * h_in + bRn) + bWn).unaryExpr(&tanh);
      Eigen::VectorXd dtdn = Eigen::VectorXd::Ones(hidden_dim) - n.cwiseProduct(n);
      Eigen::VectorXd Rnh = Rn * h_in;

      Eigen::MatrixXd inner_x;
      inner_x.array() = Wn.array() + drdx.array().colwise() * Rnh.array() + drdx.array().colwise() * bRn.array();
      Eigen::MatrixXd dndx;
      dndx.array() = inner_x.array().colwise() * dtdn.array();

      Eigen::MatrixXd inner_h;
      inner_h.array() =  Rn.array().colwise() * r.array() + drdh.array().colwise() * Rnh.array() + drdh.array().colwise() * bRn.array();
      Eigen::MatrixXd dndh;
      dndh.array() = inner_h.array().colwise() * dtdn.array();
      Eigen::MatrixXd doutdx;
      doutdx.array() = dndx.array() - dndx.array().colwise() * i.array() - didx.array().colwise() * n.array()  + didx.array().colwise() * h_in.array();
      Eigen::MatrixXd doutdh;
      doutdh.array() = dndh.array() - dndh.array().colwise() * i.array() - didh.array().colwise() * n.array() + didh.array().colwise() * h_in.array();
      doutdh.diagonal() += i;

      return {doutdx, doutdh, Eigen::MatrixXd(), Eigen::MatrixXd()};
    }
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


  /* Stacks multiple GRUs + a linear layer */
  class StackedGRUCell : public RNNCell {
  public:
    uint32_t output_dimension() const {return 0;};
    uint32_t hidden_dimension() const {return 0;};
    uint32_t input_dimension() const {return 0;};

    StackedGRUCell(int layers, int hidden_dim, std::vector<std::shared_ptr<GRUCell>>& cells, const Eigen::MatrixXd& Wdense, const Eigen::VectorXd& bdense) {
      this->layers=layers;
      this->cells=cells;
      this->hidden_dim=hidden_dim;
      this->Wdense=Wdense;
      this->bdense=bdense;
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXd> Forward(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {
      Eigen::VectorXd output = x;
      Eigen::VectorXd states = h_in;
      // gru cells
      for (int l=0; l<layers; ++l) {
	Eigen::VectorXd h_out;
	std::tie(output, h_out) = cells[l]->Forward(output, states.segment(l*hidden_dim, hidden_dim));
	states.segment(l*hidden_dim, hidden_dim)=output;
      }

      // fully connected
      output = Wdense.transpose() * output + bdense;
      return {output, states};
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {

    }

  protected:
    std::vector<std::shared_ptr<GRUCell>> cells;
    int layers;
    int hidden_dim;
    Eigen::MatrixXd Wdense;
    Eigen::VectorXd bdense;
  };
}
