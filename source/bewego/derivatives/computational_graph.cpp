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
      f_node->add_parent(f.parent);
      f.parent->add_child(f_node);
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
  BuildInputEdges();
}

void Graph::RemoveRedundantEdges() {
  for (auto edge : edges_) {
    cout << "edge 1" << endl;
  }
}

bool Graph::DoesEdgeExist(
    const std::pair<uint32_t, uint32_t>& edge,
    const std::vector<std::pair<uint32_t, uint32_t>>& edges) const {
  for (auto e : edges_) {
    if (e == edge) {
      return true;
    }
  }
  return false;
}

void Graph::BuildInputEdges() {
  input_edges_.clear();
  for (auto node : nodes_) {
    if (!node->is_atomic()) {
      auto f_n = std::dynamic_pointer_cast<const CombinationOperator>(
          node->differentiable_operator());
      for (auto f_i : f_n->input_operators()) {
        for (auto child : node->children()) {
          if (f_i.get() == child->differentiable_operator().get()) {
            auto e = std::make_pair(node->id(), child->id());
            node->add_input(child);
            input_edges_.push_back(e);
          }
        }
      }
    }
  }
}

bool Graph::IsSubGraph(std::shared_ptr<const Node> root_graph,
                       std::shared_ptr<const Graph> sub_graph) const {
  std::shared_ptr<const Node> node_sub;
  std::shared_ptr<const Node> node_graph;
  std::queue<std::shared_ptr<const Node>> sub_node_queue;
  std::queue<std::shared_ptr<const Node>> this_node_queue;
  sub_node_queue.push(sub_graph->nodes()[0]);
  this_node_queue.push(root_graph);

  while (!sub_node_queue.empty()) {
    node_sub = sub_node_queue.front();
    sub_node_queue.pop();
    node_graph = this_node_queue.front();
    this_node_queue.pop();

    for (uint32_t id = 0; id < node_sub->children().size(); id++) {
      auto child_s = node_sub->children()[id];
      auto child_g = node_graph->children()[id];

      if (child_g->type() != child_s->type()) {
        return false;
      }
      auto diff_map_g = child_g->differentiable_operator();
      auto diff_map_s = child_s->differentiable_operator();
      if (!diff_map_g->Compare(*diff_map_s)) {
        return false;
      }
      sub_node_queue.push(child_s);
      this_node_queue.push(child_g);
    }
  }
  return true;
}

DifferentiableMapPtr Graph::FindSubGraph(DifferentiableMapPtr f) const {
  std::string type = f->type();
  auto f_graph = std::make_shared<Graph>(f);
  for (auto node : nodes_) {
    if (node->type() == type) {
      if (IsSubGraph(node, f_graph)) {
        return node->differentiable_operator();
      }
    }
  }
  return DifferentiableMapPtr();
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

    auto input_nodes = s->input_nodes();
    if (std::find(input_nodes.begin(), input_nodes.end(), t) !=
        input_nodes.end()) {
      dot_str += source_id + "->" + target_id + " [color=blue]\n";
    }
  }
  dot_str += "}\n";
  return dot_str;
}