// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Phliipp Kratzer, philipp.kratzer@ipvs.uni-stuttgart.de
// rnn.cpp: Implementations of Recurrent neural network structures and
// analytical input->output Jacobians for fast computations.
#include <bewego/rnn.h>
#include <cmath>

namespace bewego {


static double Tanh(const double x)  {
      return std::tanh(x);
  }

  // ------------------------------------------------------------
  // GRUCell
  // ------------------------------------------------------------

  GRUCell::GRUCell(const Eigen::MatrixXd& Wi, const Eigen::MatrixXd& Wr, const Eigen::MatrixXd& Wn, const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rr, const Eigen::MatrixXd& Rn, const Eigen::VectorXd& bWi, const Eigen::VectorXd& bWr, const Eigen::VectorXd& bWn, const Eigen::VectorXd& bRi, const Eigen::VectorXd& bRr, const Eigen::VectorXd& bRn) {
    this->input_dim = Wi.cols();
    this->hidden_dim = Wi.rows();
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

  Eigen::VectorXd GRUCell::Forward(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {
    Eigen::VectorXd r = (Wr * x + Rr * h_in + bWr + bRr).unaryExpr(&Sigmoid);
    Eigen::VectorXd i = (Wi * x + Ri * h_in + bWi + bRi).unaryExpr(&Sigmoid);
    Eigen::VectorXd n = (Wn * x + r.cwiseProduct(Rn * h_in + bRn) + bWn).unaryExpr(&Tanh);
    Eigen::VectorXd h_out = (Eigen::VectorXd::Ones(hidden_dim) - i).cwiseProduct(n) + i.cwiseProduct(h_in);
    return h_out;
  }

  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> GRUCell::Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {
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

    Eigen::VectorXd n = (Wn * x + r.cwiseProduct(Rn * h_in + bRn) + bWn).unaryExpr(&Tanh);
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

    return {doutdx, doutdh};
  }


  // ------------------------------------------------------------
  // StackedCoupledRNNCell
  // ------------------------------------------------------------

  StackedCoupledRNNCell::StackedCoupledRNNCell(const int layers, const int hidden_dim, const int output_dim, const std::vector<std::shared_ptr<CoupledRNNCell>>& cells, const Eigen::MatrixXd& Wdense, const Eigen::VectorXd& bdense) {
    this->layers=layers;
    this->cells=cells;
    this->hidden_dim=hidden_dim;
    this->Wdense=Wdense;
    this->bdense=bdense;
    this->output_dim=output_dim;
  }

  std::tuple<Eigen::VectorXd, Eigen::VectorXd> StackedCoupledRNNCell::Forward(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {
    Eigen::VectorXd output = x;
    Eigen::VectorXd states = h_in;
    // gru cells
    for (int l=0; l<layers; ++l) {
      output = cells[l]->Forward(output, states.segment(l*hidden_dim, hidden_dim));
      states.segment(l*hidden_dim, hidden_dim)=output;
    }

    // fully connected
    output = Wdense.transpose() * output + bdense;
    return {output, states};
  }

  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> StackedCoupledRNNCell::Jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& h_in) const {
    const int input_dim = x.rows();
    Eigen::VectorXd output = x;
    Eigen::VectorXd states = h_in;
    Eigen::MatrixXd jac_h_h = Eigen::MatrixXd::Zero(layers*hidden_dim, layers*hidden_dim);
    Eigen::MatrixXd jac_h_out = Eigen::MatrixXd::Zero(output_dim, hidden_dim*layers);
    Eigen::MatrixXd jac_in_h = Eigen::MatrixXd::Zero(hidden_dim*layers, input_dim);

    for (int l=0; l<layers; ++l) {
      Eigen::MatrixXd jac_in, jac_h;
      std::tie(jac_in, jac_h) = cells[l]->Jacobian(output, states.segment(l*hidden_dim, hidden_dim));
      output = cells[l]->Forward(output, states.segment(l*hidden_dim, hidden_dim));

      if (l > 0) {
	jac_in_h.block(l*hidden_dim, 0, hidden_dim, input_dim) = jac_in * jac_in_h.block((l-1)*hidden_dim, 0, hidden_dim, input_dim);
      } else {
	jac_in_h.block(0, 0, hidden_dim, input_dim) = jac_in;
      }

      jac_h_h.block(l*hidden_dim, l*hidden_dim, hidden_dim, hidden_dim) = jac_h;

      for (int i=0; i<l; ++i) {
	jac_h_h.block(l*hidden_dim, i*hidden_dim, hidden_dim, hidden_dim) = jac_in * jac_h_h.block((l-1)*hidden_dim, i*hidden_dim, hidden_dim, hidden_dim);
      }

      states.segment(l*hidden_dim, hidden_dim)=output;
    }
    Eigen::MatrixXd jac_in_out = Wdense.transpose() * jac_in_h.block((layers-1)*hidden_dim, 0, hidden_dim, input_dim);
    for (int l=0; l<layers; ++l) {
      jac_h_out.block(0, l*hidden_dim, output_dim, hidden_dim) = Wdense.transpose() * jac_h_h.block((layers-1)*hidden_dim, l*hidden_dim, hidden_dim, hidden_dim);
    }
    return {jac_in_out, jac_in_h, jac_h_out, jac_h_h};

  }


  // ------------------------------------------------------------
  // VRED
  // ------------------------------------------------------------

  VRED::VRED(std::shared_ptr<RNNCell>& cell, int dim_trans) {
    eulertoquat = std::make_shared<EulerToQuat>();
    quattoexpmap = std::make_shared<QuatToExpmap>();
    quattoeuler = std::make_shared<QuatToEuler>();
    expmaptoquat = std::make_shared<ExpmapToQuat>();
    this->cell = cell;
    this->dim_trans = dim_trans;
  }

  Eigen::MatrixXd VRED::Forward (const Eigen::MatrixXd data, const Eigen::MatrixXd deltas, int src_length, int pred_length) const {

    Eigen::MatrixXd inputs = data;
    Eigen::MatrixXd outputs(pred_length, inputs.cols());

    // convert the rotational parts from euler to expmap
    for (int t=0; t<inputs.rows(); ++t) {
      for (int r=dim_trans; r<inputs.cols(); r+=3) {
	inputs.block(t, r, 1, 3) = quattoexpmap->Forward(eulertoquat->Forward(inputs.block(t, r, 1, 3).transpose())).transpose();
      }
    }

    // velocity inputs
    Eigen::MatrixXd vel_inputs = inputs.block(1, 0, inputs.rows()-1, inputs.cols()) - inputs.block(0, 0, inputs.rows()-1, inputs.cols());
    inputs = inputs.block(1, 0, inputs.rows()-1, inputs.cols());

    // set initial state zero
    Eigen::VectorXd states = Eigen::VectorXd::Zero(cell->hidden_dimension());

    Eigen::VectorXd cell_input(inputs.cols()*2-dim_trans);
    Eigen::VectorXd output_vel;

    // loop 1: ground truth
    for (int i=0; i<src_length-1; ++i) {
      cell_input << inputs.block(i, dim_trans, 1, inputs.cols()-dim_trans).transpose(), vel_inputs.row(i).transpose();
      std::tie(output_vel, states) = cell->Forward(cell_input, states);
    }
    Eigen::VectorXd output = inputs.row(src_length-2).transpose() + output_vel;  //residual
    outputs.row(0) = output;

    // loop 2: self conditioning
    for (int o=0; o<pred_length-1;++o) {
      cell_input << output.segment(dim_trans, output.rows() - dim_trans), output_vel + deltas.row(o).transpose();
      std::tie(output_vel, states) = cell->Forward(cell_input, states);
      output = output_vel + outputs.row(o).transpose();
      outputs.row(o+1) = output;
    }

    // convert the rotational parts from expmap to euler
    for (int t=0; t<outputs.rows(); ++t) {
      for (int r=dim_trans; r<outputs.cols(); r+=3) {
	outputs.block(t, r, 1, 3) = quattoeuler->Forward(expmaptoquat->Forward(outputs.block(t, r, 1, 3).transpose())).transpose();
      }
    }
    return outputs;
  }

  Eigen::MatrixXd VRED::Jacobian (const Eigen::MatrixXd data, const Eigen::MatrixXd deltas, int src_length, int pred_length) const {
    const int idim = cell->input_dimension();
    const int odim = cell->output_dimension();
    const int hdim = cell->hidden_dimension();

    Eigen::MatrixXd inputs = data;
    Eigen::MatrixXd outputs(pred_length, inputs.cols());
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero((pred_length-1) * odim, (pred_length-1) * odim);
    // convert the rotational parts from euler to expmap
    #pragma omp parallel for simd collapse(2)
    for (int t=0; t<inputs.rows(); ++t) {
      for (int r=dim_trans; r<inputs.cols(); r+=3) {
	inputs.block(t, r, 1, 3) = quattoexpmap->Forward(eulertoquat->Forward(inputs.block(t, r, 1, 3).transpose())).transpose();
      }
    }

    // velocity inputs
    Eigen::MatrixXd vel_inputs = inputs.block(1, 0, inputs.rows()-1, inputs.cols()) - inputs.block(0, 0, inputs.rows()-1, inputs.cols());
    inputs = inputs.block(1, 0, inputs.rows()-1, inputs.cols());

    // set initial state zero
    Eigen::VectorXd states = Eigen::VectorXd::Zero(hdim);

    Eigen::VectorXd cell_input(inputs.cols()*2-dim_trans);
    Eigen::VectorXd output_vel;

    // loop 1: ground truth
    for (int i=0; i<src_length-1; ++i) {
      cell_input << inputs.block(i, dim_trans, 1, inputs.cols()-dim_trans).transpose(), vel_inputs.row(i).transpose();
      std::tie(output_vel, states) = cell->Forward(cell_input, states);
    }
    Eigen::VectorXd output = inputs.row(src_length-2).transpose() + output_vel;  //residual
    outputs.row(0) = output;

    // loop 2: self conditioning
    Eigen::MatrixXd dhddelta((pred_length-1)*hdim, (pred_length-1)*idim);
    Eigen::MatrixXd doddelta((pred_length-1)*odim, (pred_length-1)*idim);

    for (int o=0; o<pred_length-1;++o) {
      cell_input << output.segment(dim_trans, output.rows() - dim_trans), output_vel + deltas.row(o).transpose();
      Eigen::MatrixXd states_out;
      std::tie(output_vel, states_out) = cell->Forward(cell_input, states);
      Eigen::MatrixXd jac_in_out, jac_in_h, jac_h_out, jac_h_h;
      std::tie(jac_in_out, jac_in_h, jac_h_out, jac_h_h) = cell->Jacobian(cell_input, states);
      states=states_out;

      dhddelta.block(o*hdim, o*idim, hdim, idim) = jac_in_h;
      doddelta.block(o*odim, o*idim, odim, idim) = jac_in_out;

      jacobian.block(o * odim, o * odim, odim, odim) = jac_in_out.block(0, odim-dim_trans, jac_in_out.rows(), odim);  // only second half wrt delta

      output = output_vel + outputs.row(o).transpose();
      outputs.row(o+1) = output;

      for (int d=0; d<o; ++d) {
	// jacobian of current output at timestep o wrt delta at timestep d; computed using jacobians of previous timesteps:
	Eigen::MatrixXd dod = jac_in_out.block(0, odim-dim_trans, odim, odim) * doddelta.block((o-1)*odim, d*idim, odim, idim) + jac_h_out * dhddelta.block((o-1)*hdim, d*idim, hdim, idim);
	dod += doddelta.block((o-1)*odim, d*idim, odim, idim);  //residual
	doddelta.block(o*odim, d*idim, odim, idim) = dod;
	Eigen::MatrixXd dhd = jac_in_h.block(0, odim-dim_trans, hdim, odim) * doddelta.block((o-1)*odim, d*idim, odim, idim) + jac_h_h * dhddelta.block((o-1)*hdim, d*idim, hdim, idim);
	dhddelta.block(o*hdim, d*idim, hdim, idim) = dhd;
	jacobian.block(o*odim, d*odim,odim,odim) = dod.block(0, odim-dim_trans, odim, odim);
      }
    }

    // rotational jacobian
    #pragma omp parallel for simd collapse(2)
    for (int t=1; t<outputs.rows(); ++t) {
      for (int r=dim_trans; r<outputs.cols(); r+=3) {
	Eigen::MatrixXd expmap = outputs.block(t, r, 1, 3).transpose();
	Eigen::MatrixXd quat = expmaptoquat->Forward(expmap);
	Eigen::MatrixXd jac_rot_quat = expmaptoquat->Jacobian(expmap);
	Eigen::MatrixXd jac_rot_euler = quattoeuler->Jacobian(quat);
	Eigen::MatrixXd jac_rot = jac_rot_euler * jac_rot_quat;

	for (int js=0; js<jacobian.cols(); js+=3)
	  jacobian.block(r+(t-1)*odim, js, 3, 3) = jac_rot * jacobian.block(r+(t-1)*odim, js, 3, 3);
      }
    }
    return jacobian;
  }
}
