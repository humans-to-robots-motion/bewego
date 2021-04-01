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
#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/combination_operators.h>
#include <bewego/derivatives/computational_graph.h>

#include <queue>

using namespace bewego::computational_graph;
using std::cout;
using std::endl;

void Graph::BuildFromNetwork(DifferentiableMapPtr network) {
  uint id = 0;
  nodes_.clear();
  DifferentiableMapPtr f;
  std::queue<DifferentiableMapPtr> diff_operators;
  diff_operators.push(network);
  while (!diff_operators.empty()) {
    f = diff_operators.front();
    auto node_f = std::make_shared<Node>(f, id++);
    nodes_.push_back(node_f);
    // cout << "add node : " << f->type() << endl;
    if (!f->is_atomic()) {
      auto f_c = std::dynamic_pointer_cast<const CombinationOperator>(f);
      for (auto g : f_c->nested_operators()) {
        auto node_g = std::make_shared<Node>(g, id++);
        edges_.push_back(std::make_pair(node_f->id(), node_g->id()));
        diff_operators.push(g);
      }
    }
    diff_operators.pop();
  }
}