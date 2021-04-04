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
#include <bewego/util/misc.h>

#include <queue>

using namespace bewego;
using namespace bewego::computational_graph;
using namespace bewego::util;
using std::cout;
using std::endl;

struct NodeWithParent {
  NodeWithParent() {}
  NodeWithParent(DifferentiableMapPtr f, NodePtr n) : function(f), parent(n) {}
  DifferentiableMapPtr function;
  NodePtr parent;
};

void Graph::BuildFromNetwork(DifferentiableMapPtr network) {
  int id = 0;
  nodes_.clear();
  NodeWithParent f;
  std::queue<NodeWithParent> diff_operators;
  diff_operators.push(NodeWithParent(network, NodePtr()));
  while (!diff_operators.empty()) {
    f = diff_operators.front();
    auto f_node = std::make_shared<Node>(f.function, id++);
    nodes_.push_back(f_node);
    if (f.parent) {
      edges_.push_back(std::make_pair(f.parent->id(), f_node->id()));
    }
    // cout << "add node : " << f->type() << endl;
    if (!f.function->is_atomic()) {
      auto f_c =
          std::dynamic_pointer_cast<const CombinationOperator>(f.function);
      for (auto g : f_c->nested_operators()) {
        diff_operators.push(NodeWithParent(g, f_node));
      }
    }
    diff_operators.pop();
  }
}

void Graph::RemoveRedundantEdges() {
  for (auto edge : edges_) {
    cout << "edge 1" << endl;
  }
}

void Graph::Print() const {
  cout << " -- Nb of nodes : " << nodes_.size() << endl;
  cout << " -- Nb of edges : " << edges_.size() << endl;
  for (auto edge : edges_) {
    cout << "edge : " << nodes_[edge.first]->type() << " - to - "
         << nodes_[edge.second]->type() << " : ( " << edge.first << " , "
         << edge.second << " )" << endl;
  }
}

std::string Graph::WriteToDot() const {
  std::string dot_str = "";
  dot_str += "digraph bewego_computational_graph {\n";
  for (auto edge : edges_) {
    auto s = nodes_[edge.first];
    auto t = nodes_[edge.second];
    std::string source_id = s->type() + "_" + LeftPaddingWithZeros(s->id(), 5);
    std::string target_id = t->type() + "_" + LeftPaddingWithZeros(t->id(), 5);
    dot_str += source_id + "->" + target_id + "\n";
  }
  dot_str += "}\n";
  return dot_str;
}