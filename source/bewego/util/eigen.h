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
 *                                                             Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <Eigen/Core>
#include <vector>

namespace bewego {
namespace util {

inline std::vector<double> FromEigen(const Eigen::VectorXd& v) {
  std::vector<double> vect(v.size());
  for (int i = 0; i < v.size(); i++) {
    vect[i] = v[i];
  }
  return vect;
}

template <typename Type>
inline Eigen::Matrix<Type, Eigen::Dynamic, 1> ToEigen(
    const std::vector<Type>& v) {
  Eigen::Matrix<Type, Eigen::Dynamic, 1> vect(v.size());
  for (int i = 0; i < vect.size(); i++) {
    vect[i] = v[i];
  }
  return vect;
}

//! Append two vectors.
//! [a1, ..., aN, b1, ..., bM] <-- [a1, ..., aN] + [b1, ..., bM]
template <typename Type>
inline void Append(std::vector<Type>& a, const std::vector<Type>& b) {
  a.insert(std::end(a), std::begin(b), std::end(b));
}

Eigen::MatrixXd HStack(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
Eigen::MatrixXd VStack(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

/**
 Serialize to string the difference objects provided as input
*/
class Serializer {
 public:
  Serializer() : use_scientific_(false), full_precision_(false) {}
  std::string Serialize(const Eigen::VectorXd& v);
  std::string Serialize(const Eigen::MatrixXd& v);

 private:
  bool use_scientific_;
  bool full_precision_;
};

//! Deserialize a Matrix from a string
//! asumes a column major encoding
//! TODO test for matrices
Eigen::MatrixXd FromString(const std::string& matrix_txt, int rows, int cols);

//! Serialize a vector to string
//! Todo test
std::string ToString(const Eigen::MatrixXd& v, bool use_scientific_csv,
                     bool full_precision_csv);

}  // namespace util
}  // namespace bewego
