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
 *                                                             Thu 1 Apr 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <bewego/derivatives/differentiable_map.h>

namespace bewego {
namespace computational_graph {

class Node {
 public:
  Node(DifferentiableMapPtr atomic_operator, uint32_t id)
      : atomic_operator_(atomic_operator), id_(id) {}

  uint32_t id() const { return id_; }
  std::string type() const { return atomic_operator_->type(); }
  DifferentiableMapPtr atomic_operator() const { return atomic_operator_; }

 protected:
  DifferentiableMapPtr atomic_operator_;
  uint32_t id_;
};

class Graph {
 public:
  Graph() {}
};

}  // namespace computational_graph
};  // namespace bewego