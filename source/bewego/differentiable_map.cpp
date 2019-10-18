#include <bewego/differentiable_map.h>

namespace bewego {

Eigen::MatrixXd DifferentiableMap::FiniteDifferenceJacobian(
        const DifferentiableMap& f,
        const Eigen::VectorXd& q)
        {
        assert(q.size() == f.input_dimension());
        double dt = 1e-4;
        double dt_half = dt / 2.;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero( 
            f.output_dimension(),
            f.input_dimension());
        for (uint32_t j=0; j<q.size(); j++) {
            
            Eigen::VectorXd q_up = q;
            q_up[j] += dt_half;
            Eigen::VectorXd x_up = f(q_up);

            Eigen::VectorXd q_down = q;
            q_down[j] -= dt_half;
            Eigen::VectorXd x_down = f(q_down);
            
            J.col(j) = (x_up - x_down) / dt;
        }
        return J;
}

/**
Takes an object f that has a forward method returning
    a numpy array when querried.
    */
Eigen::MatrixXd DifferentiableMap::FiniteDifferenceHessian(
        const DifferentiableMap& f,
        const Eigen::VectorXd& q) {
        assert(q.size() == f.input_dimension());
        assert(f.output_dimension() == 1);
        double dt = 1e-4;
        double dt_half = dt / 2.;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(
            f.input_dimension(),
            f.input_dimension());
        for (uint32_t j=0; j<q.size(); j++) {
            
            Eigen::VectorXd q_up = q;
            q_up[j] += dt_half;
            Eigen::VectorXd g_up = f.Gradient(q_up);

            Eigen::VectorXd q_down = q;
            q_down[j] -= dt_half;
            Eigen::VectorXd g_down = f.Gradient(q_down);

            H.col(j) = (g_up - g_down) / dt;
        }
        return H;
    }

} // namespace bewego