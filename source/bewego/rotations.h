#pragma once
#include <bewego/differentiable_map.h>
#include <iostream>

namespace bewego {

  class QuatToEuler : public DifferentiableMap {
  public:
    QuatToEuler() {}

    uint32_t output_dimension() const { return 3; }
    uint32_t input_dimension() const { return 4; }

    /** q - quaternion [x, y, z, w] **/
    Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
      assert(q.size() == 4);
      const double& x = q[0];
      const double& y = q[1];
      const double& z = q[2];
      const double& w = q[3];
      double t0 = 2. * (w * x + y * z);
      double t1 = 1. - 2. * (x*x + y*y);
      double X = atan2(t0, t1);
      double t2 = 2. * (w*y - z*x);

      t2 = fmin(.99999, t2);
      t2 = fmax(-.99999, t2);

      double Y = asin(t2);

      double t3 = 2. * (w * z + x * y);
      double t4 = 1. - 2. * (y*y + z*z);
      double Z = atan2(t3, t4);

      Eigen::VectorXd euler(3);
      euler << X, Y, Z;
      return euler;
    }

    Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
      assert(q.size() == 4);
      const double& x = q[0];
      const double& y = q[1];
      const double& z = q[2];
      const double& w = q[3];
      double t0 = 2. * (w * x + y * z);
      Eigen::Vector4d dt0(2.*w, 2.*z, 2.*y, 2.*x);
      double t1 = 1. - 2. * (x*x + y*y);
      Eigen::Vector4d dt1(-4.*x, -4.*y, 0., 0.);

      Eigen::Vector4d dX = -t0/(t0*t0+t1*t1)*dt1 + t1/(t0*t0+t1*t1)*dt0;

      double t2 = 2. * (w*y - z*x);
      Eigen::Vector4d dY;

      if (t2 < -.99999 || t2 > .99999) {
	dY << 0., 0., 0., 0.;
      } else {
	Eigen::Vector4d dt2;
	dt2 << -2.*z, 2.*w, -2.*x, 2.*y;
	dY = 1./(sqrt(1. - t2*t2)) * dt2;
      }


      double t3 = 2. * (w * z + x * y);
      double t4 = 1. - 2. * (y*y + z*z);
      Eigen::Vector4d dt3(2.*y, 2.*x, 2.*w, 2.*z);
      Eigen::Vector4d dt4(0., -4.*y, -4.*z, 0.);
      Eigen::Vector4d dZ = -t3/(t3*t3+t4*t4)*dt4 + t4/(t3*t3+t4*t4)*dt3;
      Eigen::MatrixXd jac(3, 4);
      jac.row(0) = dX;
      jac.row(1) = dY;
      jac.row(2) = dZ;
      return jac;
    }
  };

  class ExpmapToQuat : public DifferentiableMap {
  public:
    ExpmapToQuat() {}

    uint32_t output_dimension() const { return 4; }
    uint32_t input_dimension() const { return 3; }

    double sinc(const double x) const {
      if (x == 0)
	return 1;
      return sin(x)/x;
    }

    Eigen::VectorXd Forward(const Eigen::VectorXd& e) const {
      assert(e.size() == 3);
      double theta = e.norm();
      double w = cos(.5*theta);
      Eigen::Vector3d xyz = .5 * sinc(.5*theta) * e;
      return Eigen::Vector4d(xyz[0], xyz[1], xyz[2], w);

    }

    Eigen::MatrixXd Jacobian(const Eigen::VectorXd& e) const {
      assert(e.size() == 3);
      double theta = e.norm();
      Eigen::Vector3d dtheta = (1. / theta) * e;
      Eigen::Vector3d dw = dtheta * .5 * -sin(.5*theta);
      Eigen::Vector3d xyz = .5 * sinc(.5*theta) * e;
      double dsinc = .5 * (theta*cos(0.5*theta) - 2*sin(0.5*theta))/(theta*theta);
      Eigen::MatrixXd dot = e * dtheta.transpose() * dsinc;
      Eigen::MatrixXd dxyz = dot + .5 * sinc(.5*theta) * Eigen::MatrixXd::Identity(3,3);
      Eigen::MatrixXd jac(4, 3);
      jac.block(0, 0, 3, 3) = dxyz;
      jac.row(3) = dw;
      return jac;
    }
  };

  class EulerToQuat : public DifferentiableMap {
  public:
    EulerToQuat() {}

    uint32_t output_dimension() const { return 4; }
    uint32_t input_dimension() const { return 3; }

    Eigen::VectorXd Forward(const Eigen::VectorXd& e) const {
      assert(e.size() == 3);
      const double& x = e[0];
      const double& y = e[1];
      const double& z = e[2];
      double c1 = cos(x / 2.);
      double c2 = cos(y / 2.);
      double c3 = cos(z / 2.);
      double s1 = sin(x / 2.);
      double s2 = sin(y / 2.);
      double s3 = sin(z / 2.);
      double xq = s1*c2*c3 - c1*s2*s3;
      double yq = s1*c2*s3 + c1*s2*c3;
      double zq = c1*c2*s3 - s1*s2*c3;
      double wq = c1*c2*c3 + s1*s2*s3;
      return Eigen::Vector4d(xq, yq, zq, wq);
    }
  };

  class QuatToExpmap : public DifferentiableMap {
  public:
    QuatToExpmap() {}

    uint32_t output_dimension() const { return 3; }
    uint32_t input_dimension() const { return 4; }

    Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
      assert(q.size() == 4);
      Eigen::Vector3d qxyz(q[0], q[1], q[2]);
      double qw = q[3];
      double norm = qxyz.norm();
      qxyz/=norm;
      double theta = 2. * atan2(norm, qw);
      theta = fmod(theta + 2. * M_PI, 2. * M_PI);
      return qxyz * theta;
    }
  };

}
