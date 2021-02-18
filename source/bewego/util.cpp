#include<bewego/util.h>

namespace bewego {
namespace util {

/// Real \in [0, 1]
double Rand() { return std::rand() / double(RAND_MAX); }

/// Vector \in [0, 1]^dim
Eigen::VectorXd Random(uint32_t dim) {
  return 0.5 * (Eigen::VectorXd::Random(dim) + Eigen::VectorXd::Ones(dim));
}

/// Samples a random vector using eigen's interface for sampling
/// set the seed with: std::srand((unsigned int) time(0));
Eigen::VectorXd RandomVector(uint32_t dim, double min, double max) {
  return 0.5 * ((max - min) * Eigen::VectorXd::Random(dim) +
                (max + min) * Eigen::VectorXd::Ones(dim));
}

}
}