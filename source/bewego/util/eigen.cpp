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

#include <bewego/util/eigen.h>
#include <bewego/util/misc.h>

using std::cout;
using std::endl;

namespace bewego {
namespace util {

Eigen::MatrixXd HStack(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  Eigen::MatrixXd C(A.rows(), A.cols() + B.cols());
  C << A, B;
  return C;
}

Eigen::MatrixXd VStack(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  // eigen uses provided dimensions in declaration to determine
  // concatenation direction
  Eigen::MatrixXd D(A.rows() + B.rows(),
                    A.cols());  // <-- D(A.rows() + B.rows(), ...)
  // <-- syntax is the same for vertical and horizontal concatenation
  D << A, B;
  return D;
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& s) {
  for (uint32_t i = 0; i < s.size(); i++) {
    os << s[i] << " ";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<double>& v) {
  for (uint32_t i = 0; i < v.size(); i++) {
    os << v[i] << " ";
  }
  return os;
}

Eigen::MatrixXd FromString(const std::string& matrix_txt, int rows, int cols) {
  // cout << "matrix_txt.length() : " << matrix_txt.length() << endl;
  // cout << "matrix_txt : " << matrix_txt << endl;
  std::vector<std::string> stringRows = ParseCsvString2(matrix_txt, "\n");
  std::vector<std::string> stringVector;
  for (uint32_t i = 0; i < stringRows.size(); i++) {
    std::string row = stringRows[i];
    std::vector<std::string> stringRow = ParseCsvString2(row, ",");
    // cout << "row[" << i << "] : " << stringRow << endl;
    stringVector.insert(stringVector.end(), stringRow.begin(), stringRow.end());
  }
  stringVector.resize(rows * cols);
  // cout << "stringVector : " << stringVector << endl;
  std::vector<double> doubleVector(stringVector.size());
  std::transform(stringVector.begin(), stringVector.end(), doubleVector.begin(),
                 [](const std::string& val) { return std::stod(val); });
  // cout << "doubleVector : " << doubleVector << endl;
  return Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      &doubleVector.data()[0], rows, cols);
}

std::string ToString(const Eigen::MatrixXd& v, bool use_scientific_csv,
                     bool full_precision_csv) {
  Eigen::IOFormat CSVFormat;
  if (full_precision_csv) {
    CSVFormat =
        Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  } else {
    CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                ", ", "\n");
  }
  std::stringstream ss;
  if (use_scientific_csv) {
    ss << std::scientific << v.format(CSVFormat) << endl;
  } else {
    ss << v.format(CSVFormat) << endl;
  }

  return ss.str();
}

std::string Serializer::Serialize(const Eigen::VectorXd& v) {
  std::string str("");
  str += "vector\n";
  str += "size:" + std::to_string(v.size()) + "\n";
  str += ToString(v, use_scientific_, full_precision_);
  return str;
}
std::string Serializer::Serialize(const Eigen::MatrixXd& m) {
  std::string str("");
  str += "matrix\n";
  str += "rows:" + std::to_string(m.rows()) + "\n";
  str += "cols:" + std::to_string(m.cols()) + "\n";
  str += ToString(m, use_scientific_, full_precision_);
  return str;
}

}  // namespace util
}  // namespace bewego