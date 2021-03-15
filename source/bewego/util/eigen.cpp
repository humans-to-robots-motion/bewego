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

Eigen::MatrixXd FromString(const std::string& matrix_txt, int rows, int cols) {
  std::vector<std::string> stringRows = ParseCsvString2(matrix_txt, "\n");
  std::vector<std::string> stringVector;
  for (uint32_t i = 0; i < stringRows.size(); i++) {
    std::string row = stringRows[i];
    std::vector<std::string> stringRow = ParseCsvString2(row, ",");
    Append(stringVector, stringRow);
  }
  stringVector.resize(rows * cols);
  std::vector<double> doubleVector(stringVector.size());
  std::transform(stringVector.begin(), stringVector.end(), doubleVector.begin(),
                 [](const std::string& val) { return std::stod(val); });
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

// -----------------------------------------------------------------------------
// Serializer Implementation
// -----------------------------------------------------------------------------

std::string Serializer::Serialize(const Eigen::VectorXd& v) const {
  std::string str("");
  str += "vector\n";
  str += "size:" + std::to_string(v.size()) + "\n";
  str += ToString(v, use_scientific_, full_precision_);
  return str;
}

std::string Serializer::Serialize(const Eigen::MatrixXd& m) const {
  std::string str("");
  str += "matrix\n";
  str += "rows:" + std::to_string(m.rows()) + "\n";
  str += "cols:" + std::to_string(m.cols()) + "\n";
  str += ToString(m, use_scientific_, full_precision_);
  return str;
}

Eigen::MatrixXd Serializer::Deserialize(const std::string& str) const {
  std::vector<std::string> tokens = ParseCsvString2(str, "\n", 3);
  if (tokens.size() != 3) {
    throw std::runtime_error("Size of serialized matrix < 3");
  }
  std::string type = tokens[0];
  std::string rows = tokens[1].substr(5, tokens[1].length() - 5);
  std::string cols = tokens[2].substr(5, tokens[2].length() - 5);
  int nrows = std::stod(rows);
  int ncols = std::stod(cols);
  if (nrows <= 0 || ncols <= 0) {
    throw std::runtime_error("Eigen matrix can not deserialize");
  }
  size_t start =
      tokens[0].length() + tokens[1].length() + tokens[2].length() + 3;
  std::string matrix_str = str.substr(start, str.length() - start);
  return FromString(matrix_str, nrows, ncols);
}

}  // namespace util
}  // namespace bewego