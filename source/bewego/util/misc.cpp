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
#include <bewego/util/misc.h>

#include <fstream>

// Only works on linux...
#include <unistd.h>
#include <iomanip>
#include <atomic>

using std::cerr;
using std::cout;
using std::endl;

namespace bewego {
namespace util {

// If this is set to true CSV save will use scientific notation
// We allow to set it to false be backward compatible.
static bool use_scientific_csv = true;
static bool full_precision_csv = true;

// Todo check if it is faster than ros::package
// static std::string rieef_data_ = "/usr/local/jim_bkp/rieef_data/";
// void SetRieefData() { rieef_data_ = ros::package::getPath("rieef_data"); }

// std::string GetTmpDataDirectory() {
//   return ros::package::getPath("rieef_utils") + "/data/";
// }

// std::string GetRieefDataDirectory() {
//   return ros::package::getPath("rieef_data");
// }

std::string GetCurrentDirectory() {
  char buff[FILENAME_MAX];
  char* success = getcwd(buff, FILENAME_MAX);
  std::string current_working_dir(buff);
  return current_working_dir;
}

void SetScientificCSV(bool v) { use_scientific_csv = v; }

// Parse string as tokens
std::vector<std::string> ParseCsvString(const std::string& str,
                                        bool trim_tokens) {
  std::vector<std::string> parsed_values;
  std::string string_copy = str;
  char* p = const_cast<char*>(string_copy.c_str());
  p = std::strtok(p, ",");
  while (p) {
    cout << "Token : " << p << endl;
    parsed_values.push_back(p);
    p = std::strtok(NULL, ",");
  }
  return parsed_values;
}

std::string LeftPaddingWithZeros(uint32_t id, uint32_t nb_zeros) {
  // Get id string filled with 3 zeros
  std::stringstream ss;
  ss.str("");
  ss << std::setw(nb_zeros) << std::setfill('0') << id;
  return ss.str();
}

bool SaveStringOnDisk(const std::string& filename, const std::string& txt) {
  std::ofstream file(filename.c_str(), std::ofstream::out);
  if (file.is_open()) {
    file << txt << endl;
    file.close();
    return true;
  } else {
    return false;
  }
}

int uint_to_int(uint32_t v) {
  if (v > std::numeric_limits<int>::max())
    throw std::overflow_error("Invalid cast.");
  return static_cast<int>(v);
}

uint32_t size_t_to_uint(size_t v) {
  if (v > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("Invalid cast.");
  return static_cast<uint32_t>(v);
}

uint32_t size_t_to_uint(long v) {
  if (v > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("Invalid cast.");
  return static_cast<uint32_t>(v);
}

uint32_t float_to_uint(double v) {
  auto data = std::lround(v);
  if (data > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("Invalid cast.");
  return static_cast<uint32_t>(data);
}

constexpr unsigned int str2int(const char* str, int h) {
  return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

std::vector<std::string> ListDirectory(std::string directory,
                                       std::string extension,
                                       int nb_max_files) {
  bool quiet = true;
  std::vector<std::string> files;

  std::string command = "ls " + directory;
  FILE* fp = popen(command.c_str(), "r");
  if (fp == NULL) {
    cout << "ERROR in system call" << endl;
    return files;
  }
  char str[PATH_MAX];
  while (fgets(str, PATH_MAX, fp) != NULL) {
    std::string filename(str);
    filename = filename.substr(0, filename.size() - 1);
    std::string file_extension(filename.substr(filename.find_last_of(".") + 1));
    if (extension == file_extension) {
      if (!quiet) {
        cout << "add : " << filename << endl;
      }
      files.push_back(filename);
    }

    if ((nb_max_files > 0) && (int(files.size()) >= nb_max_files)) {
      break;
    }
  }
  pclose(fp);
  return files;
}

std::string RemoveExtension(const std::string& filename) {
  size_t lastdot = filename.find_last_of(".");
  if (lastdot == std::string::npos) return filename;
  return filename.substr(0, lastdot);
}

// general case, stream interface
inline size_t word_count(std::stringstream& is)
// can pass an open std::ifstream() to this if required
{
  //    cout << is.str() << endl;
  size_t c = 0;
  for (std::string w; std::getline(is, w, ','); ++c)
    ;
  //        cout << "found word : " << w << endl;
  return c;
}

// simple string interface
inline size_t word_count(const std::string& str) {
  //    cout << "line is : " << str << endl;
  std::stringstream ss(str);
  return word_count(ss);
}

Eigen::VectorXd ReadRowVectorFromCsvFile(std::string filepath) {
  Eigen::VectorXd vector;
  std::ifstream file(filepath.c_str());
  if (!file) {
    cerr << "WARNING -- Could not read vector from file: " << filepath << endl;
    return vector;
  }
  std::string line;
  std::getline(file, line);
  uint32_t n = word_count(line);
  vector.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    file >> vector(i);
  }
  file.close();
  return vector;
}

Eigen::MatrixXd ReadMatrixFromCsvFile(std::string filepath) {
  Eigen::MatrixXd matrix;
  //    cout << "load matrix from : " << filename << endl;

  std::ifstream file(filepath.c_str(), std::ifstream::in);

  if (file.good() && file.is_open()) {
    std::string line;
    std::string cell;

    int n_rows = std::count(std::istreambuf_iterator<char>(file),
                            std::istreambuf_iterator<char>(), '\n');

    int i = 0, j = 0;

    file.clear();
    file.seekg(0, std::ios::beg);

    while (file.good()) {
      std::getline(file, line);
      std::stringstream lineStream(line);

      if (i == 0) {
        int n_cols = word_count(line);
        matrix = Eigen::MatrixXd(n_rows, n_cols);
        //                cout << "size : ( "
        // << n_rows << " , " << n_cols << " )" << endl;
      }

      j = 0;

      while (std::getline(lineStream, cell, ',')) {
        util::convert_text_to_num<double>(matrix(i, j), cell, std::dec);
        j++;
      }
      i++;
    }

    file.close();
  } else {
    cout << "could not open file : " << filepath << endl;
  }

  return matrix;
}

bool SaveMatrixToCsvFile(std::string filename, const Eigen::MatrixXd& mat) {
  Eigen::IOFormat CSVFormat;
  if (full_precision_csv) {
    CSVFormat =
        Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  } else {
    CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                ", ", "\n");
  }
  std::ofstream file(filename.c_str(), std::ofstream::out);
  if (file.is_open()) {
    if (use_scientific_csv) {
      file << std::scientific << mat.format(CSVFormat) << endl;
    } else {
      file << mat.format(CSVFormat) << endl;
    }
    file.close();
    return true;
  } else {
    return false;
  }
}

/// General interface to save matrices to file
/// Returns true if success
bool SaveMatrixToDisk(const std::string& filename, const Eigen::MatrixXd& mat,
                      bool binary) {
  if (binary) {
    return util::SaveMatrixBinary(filename.c_str(), mat);
  } else {
    return SaveMatrixToCsvFile(filename, mat);
  }
}

/// General interface to load matrices from file
bool LoadMatrixFromDisk(const std::string& filename, Eigen::MatrixXd* mat,
                        bool binary) {
  bool success = true;
  if (binary) {
    success = util::ReadMatrixBinary(filename.c_str(), *mat);
  } else {
    (*mat) = ReadMatrixFromCsvFile(filename);
    if (mat->rows() == 0 || mat->cols() == 0) {
      success = false;
    }
  }
  return success;
}

/// Real \in [0, 1]
double Rand() { return std::rand() / double(RAND_MAX); }

/// Real \in [min, max]
double RandUniform(double min, double max) {
  assert(max > min);
  return (max - min) * Rand() + min;
}

/// Vector \in [0, 1]^dim
Eigen::VectorXd Random(uint32_t dim) {
  return 0.5 * (Eigen::VectorXd::Random(dim) + Eigen::VectorXd::Ones(dim));
}

Eigen::VectorXd RandomVector(uint32_t dim, double min, double max) {
  return 0.5 * ((max - min) * Eigen::VectorXd::Random(dim) +
                (max + min) * Eigen::VectorXd::Ones(dim));
}

//! Get samples in [-0.5, 0.5]^N box
std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd> >
SampleStartAndEndConfiguration(uint32_t nb_samples, uint32_t dim) {
  Eigen::VectorXd x_max = 0.5 * Eigen::VectorXd::Ones(dim);
  Eigen::VectorXd x_min = -0.5 * Eigen::VectorXd::Ones(dim);

  Eigen::VectorXd x_init, x_goal;
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd> > pairs;
  for (int i = 0; i < nb_samples; i++) {
    x_init = (x_max - x_min).array() * Random(dim).array() + x_min.array();
    x_goal = (x_max - x_min).array() * Random(dim).array() + x_min.array();
    pairs.push_back(std::make_pair(x_init, x_goal));
  }
  return pairs;
}

const double max_allowed = 30;  // 30 : exp(30) = 10, 686, 474, 581, 524.00
const double exp_offset = 0;
void ExponentiateMatrix(Eigen::MatrixXd& values) {
  for (int i = 0; i < values.size(); i++) {
    if (values(i) > max_allowed) {
      values(i) = max_allowed;
    }
    // Add 1e-5 otherwise can not invert.
    values(i) = std::exp(*(values.data() + i)) + exp_offset;
  }
}

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

void PrintFormatedVector(const std::string& name, const Eigen::VectorXd& v) {
  cout << std::setprecision(3) << name << " : " << v.transpose() << endl;
}

static const bool no_progress_bar = true;
static std::atomic_uint printed;

void PrintProgressBarSimple(double progress) {
  if (progress < 0) {
    return;
  }
  uint32_t percentage = uint32_t(100 * progress);
  // Only valid if is called at .20, .40, .60, .80, 1
  if (percentage % 20 == 0) {
    if (printed < percentage) {
      cout << percentage << "% porcessing ..." << endl;
      printed = percentage;
    }
  }
  if (percentage == 0) {
    printed = 0;
  }
}

void PrintProgressBar(double progress) {
  if (no_progress_bar) {
    PrintProgressBarSimple(progress);
    return;
  }

  const int barWidth = 70;
  int pos = barWidth * progress;
  std::cout << "[";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();
}

void print_seed(int seed) {
  std::cout << "MultivariateGaussian seed : " << seed << std::endl;
}

}  // namespace util
}  // namespace bewego
