// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <stdexcept>
#include <vector>

namespace bewego {
namespace util {

template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop, IntType step) {
  if (step == IntType(0)) {
    throw std::invalid_argument("step for range must be non-zero");
  }

  std::vector<IntType> result;
  IntType i = start;
  while ((step > 0) ? (i < stop) : (i > stop)) {
    result.push_back(i);
    i += step;
  }

  return result;
}

template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop) {
  return range(start, stop, IntType(1));
}

template <typename IntType>
std::vector<IntType> range(IntType stop) {
  return range(IntType(0), stop, IntType(1));
}

}  // namespace util
}  // namespace bewego