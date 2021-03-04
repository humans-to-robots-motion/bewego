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
#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <vector>

namespace bewego {
namespace util {

#define db_print(obj) \
  std::cout << "[ " #obj " ]: " << std::flush << (obj) << std::endl

#define db_mat_print(obj)                                                 \
  std::cout << "------------------------------------------" << std::endl; \
  std::cout << "[ " #obj " ]: " << std::endl                              \
            << (obj.format(Eigen::IOFormat(4, 0, ", ", "\n", "[", "]")))  \
            << std::endl;                                                 \
  std::cout << "------------------------------------------" << std::endl

/// TODO use c++11 function instead
/// std::stoi or std::stod
template <class T>
bool convert_text_to_num(T& t,
                         const std::string& s,
                         std::ios_base& (*f)(std::ios_base&)) {
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
}

// has nan
template <typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& m) {
  for (int i = 0; i < m.rows(); i++)
    for (int j = 0; j < m.cols(); j++) {
      if (std::isnan(m(i, j))) return true;
    }
  return false;
}

//! Get the tmp directory for testing rieef utilities
// std::string GetTmpDataDirectory();

//! Get the data directory that is specified using a environment variable
// std::string GetRieefDataDirectory();

//! WTF still not part of the standard...
std::string GetCurrentDirectory();

// Simple wrapper to write in a simple txt file.
// Adds a line after the string.
bool SaveStringOnDisk(const std::string& filename, const std::string& txt);

// Saves an Eigen matrix in the most compact way.
// This is not human readable.
template <class Matrix>
bool SaveMatrixBinary(const char* filename, const Matrix& matrix) {
  std::ofstream out(filename,
                    std::ios::out | std::ios::binary | std::ios::trunc);
  bool success = false;
  if (out.is_open()) {
    typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
    out.write((char*)(&rows), sizeof(typename Matrix::Index));
    out.write((char*)(&cols), sizeof(typename Matrix::Index));
    out.write((char*)matrix.data(),
              rows * cols * sizeof(typename Matrix::Scalar));
    out.close();
    success = true;
  }
  return success;
}

template <class Matrix>
bool ReadMatrixBinary(const char* filename, Matrix& matrix) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  bool success = false;
  if (in.is_open()) {
    typename Matrix::Index rows = 0, cols = 0;
    in.read((char*)(&rows), sizeof(typename Matrix::Index));
    in.read((char*)(&cols), sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read((char*)matrix.data(),
            rows * cols * sizeof(typename Matrix::Scalar));
    in.close();
    success = true;
  }
  return success;
}

// Parse string as tokens
std::vector<std::string> ParseCsvString(const std::string& str,
                                        bool trim_tokens = false);

//! Fill with zeros.
std::string LeftPaddingWithZeros(uint32_t id, uint32_t nb_zeros = 3);

//! List files in a directory
std::vector<std::string> ListDirectory(std::string directory,
                                       std::string extension,
                                       int nb_max_files = -1);

//! Remove extension from filename
std::string RemoveExtension(const std::string& filename);

/// Load Eigen matrix
/// Compatible with the matlab csv format, can be used for vectors
Eigen::MatrixXd ReadMatrixFromCsvFile(std::string filename);

///! Load vector from file where the vector is stored as a
/// row matrix of values, in the general case use the matrix
/// function and cast to vector.
Eigen::VectorXd ReadRowVectorFromCsvFile(std::string filepath);

// Set CSV save to scientific.
void SetScientificCSV(bool v);

/// save eigen matrix
bool SaveMatrixToCsvFile(std::string filename, const Eigen::MatrixXd& mat);

/// General interface to save matrices to file
bool SaveMatrixToDisk(const std::string& filename,
                      const Eigen::MatrixXd& mat,
                      bool binary);

/// General interface to load matrices from file
bool LoadMatrixFromDisk(const std::string& filename,
                        Eigen::MatrixXd* mat,
                        bool binary);

/// Float \in [0, 1]
/// set the seed with: std::srand((unsigned int) time(0));
double Rand();

/// Float \in [min, max]
/// set the seed with: std::srand((unsigned int) time(0));
double RandUniform(double min, double max);

/// Vector \in [0, 1]
/// set the seed with: std::srand((unsigned int) time(0));
Eigen::VectorXd Random(uint32_t dim);

/// Samples a random vector using eigen's interface for sampling
/// set the seed with: std::srand((unsigned int) time(0));
Eigen::VectorXd RandomVector(uint32_t size, double min, double max);

//! Get samples in [-0.5, 0.5]^N box
std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>
SampleStartAndEndConfiguration(uint32_t nb_samples, uint32_t dim);

// Exponentiate matrix with internal max value.
void ExponentiateMatrix(Eigen::MatrixXd& values);

//! Casts a whole vector of shared pointers to const shared pointers
template <typename Derived>
std::vector<std::shared_ptr<const Derived>> ConvertToConst(
    const std::vector<std::shared_ptr<Derived>>& vector_not_const) {
  std::vector<std::shared_ptr<const Derived>> vector_const;
  for (auto ptr : vector_not_const) {
    vector_const.push_back(std::const_pointer_cast<const Derived>(ptr));
  }
}

// proper convert
int uint_to_int(uint32_t v);
uint32_t size_t_to_uint(size_t data);
uint32_t size_t_to_uint(long data);
uint32_t float_to_uint(double v);
constexpr unsigned int str2int(const char* str, int h = 0);

// Convert a vector of trajectories to a vector of shared pointer
template <typename Type>
inline std::vector<std::shared_ptr<const Type>> ConvertToSharedPtr(
    const std::vector<Type>& v) {
  std::vector<std::shared_ptr<const Type>> trajectories;
  for (auto t : v) {
    trajectories.push_back(std::make_shared<const Type>(t));
  }
  return trajectories;
}

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

template <typename Integer>
std::vector<Integer> Range(Integer start, Integer end, Integer step = 1) {
  std::vector<Integer> indicies(end - start);
  uint32_t k = 0;
  for (int i = start; i < end; i += step) {
    indicies[k++] = i;
  }
  return indicies;
}

//! Append two vectors.
//! [a1, ..., aN, b1, ..., bM] <-- [a1, ..., aN] + [b1, ..., bM]
template <typename Type>
inline std::vector<Type> AppendConst(const std::vector<Type>& a,
                                     const std::vector<Type>& b) {
  std::vector<Type> c = a;
  c.insert(std::end(c), std::begin(b), std::end(b));
  return c;
}

//! Append two vectors.
//! [a1, ..., aN, b1, ..., bM] <-- [a1, ..., aN] + [b1, ..., bM]
template <typename Type>
inline void Append(std::vector<Type>& a, const std::vector<Type>& b) {
  a.insert(std::end(a), std::begin(b), std::end(b));
}

Eigen::MatrixXd HStack(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
Eigen::MatrixXd VStack(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

//! Prints formated vector.
void PrintFormatedVector(const std::string& name, const Eigen::VectorXd& v);

//! Prints a progress bar by flushing the stream.
void PrintProgressBar(double progress);

//! Function to print seed
void print_seed(int seed);
}
}  // namespace bewego
