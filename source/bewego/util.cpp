#include<bewego/util.h>

namespace bewego {
namespace util {

/// Real \in [0, 1]
double Rand() { return std::rand() / double(RAND_MAX); }

/// Vector \in [0, 1]^dim
Eigen::VectorXd Random(uint32_t dim) {
  return 0.5 * (Eigen::VectorXd::Random(dim) + Eigen::VectorXd::Ones(dim));
}

}
}