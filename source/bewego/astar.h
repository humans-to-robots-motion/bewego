/*
 * Copyright (c) 2019
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
/*
 * A_Star.h
 *
 * This header file contains declerations of four classes
 *
 * 1) Tree_Element
 * 2) Queue_Element
 * 3) Prioritize_Queue_Elements
 * 4) A_Star
 *
 * */

#pragma once

#include <queue>
#include <vector>

namespace bewego {

class SearchState {
 public:
  SearchState();
  double Cost(SearchState* parent, SearchState* goal);  // f (f=g+h)

  double f() const;  // get functions for the variables
  double g() const;
  double h() const;

  // Test that the state is valid
  virtual bool valid() { return true; }

  /* states after branching is returned and the number of
   non-NULL states in the returned array is saved in the variable nodes_n */
  virtual std::vector<SearchState*> Successors(SearchState* s);
  virtual bool leaf(); /* leaf control for an admissible heuristic function;
                            the test of h==0*/
  virtual bool equal(SearchState* other);

  virtual void set_closed(std::vector<SearchState*>& closed_states,
                          std::vector<SearchState*>& open_states);
  virtual void set_open(std::vector<SearchState*>& open_states);
  
  virtual bool is_closed(std::vector<SearchState*>& closed_states);
  virtual bool is_open(std::vector<SearchState*>& open_states);

  virtual void reset() {}
  virtual void print() {}

 protected:
  virtual double Length(SearchState* parent);                // g
  virtual double Heuristic(SearchState* p, SearchState* g);  // h

 private:
  // f, g, h values
  double f_, g_, h_;
};

/**
 * @ingroup SEARCH
 * This class is the node class
 * to implement the search tree
 */
class TreeNode {
 public:
  TreeNode() : parent_(NULL) {}
  TreeNode(bewego::SearchState*, TreeNode*);
  ~TreeNode() {}

  TreeNode* parent() const { return parent_; }
  bewego::SearchState* search_state() const { return search_state_; }

 private:
  TreeNode* parent_;
  bewego::SearchState* search_state_;
};

/**
 * @ingroup SEARCH
 * Basic block to be used in
 * the priority queue.
 */
class QueueElement {
 public:
  QueueElement() : node_(NULL) {}
  QueueElement(TreeNode* n) : node_(n) {}
  ~QueueElement() {}

  TreeNode* tree_node() { return node_; }

  friend class PrioritizeQueueElements;

 private:
  TreeNode* node_;
};

/**
 * @ingroup SEARCH
 * Function used for sorting tree nodes
 * in the priority queue
 */
class PrioritizeQueueElements {
 public:
  int operator()(QueueElement& x, QueueElement& y) {
    return x.tree_node()->search_state()->f() >
           y.tree_node()->search_state()->f();
  }
};

/**
 * @ingroup SEARCH
 * @brief This class keeps a pointer to the A-star search tree, an instant
 *  of priority_queue of "Queue_Element"s. Solve returns a vector of
 *  states
 */
class AStar {
 public:
  AStar() : goal_(NULL), goal_is_defined_(false) {}
  AStar(bewego::SearchState* goal) : goal_(goal), goal_is_defined_(true) {}
  ~AStar() {}

  std::vector<SearchState*> Solve(SearchState* initial_state);

 private:
  void set_goal(SearchState* v) { goal_ = v; }
  bool is_goal(SearchState* state);
  std::vector<SearchState*> Solution(QueueElement qEl);
  void CleanSearchStates();

  bool goal_is_defined_;     // true if goal is defined
  SearchState* goal_;        // goal search state
  TreeNode* root_;           // root of the A-star tree
  TreeNode* solution_leaf_;  // keeps the solution leaf after solve is called

  std::priority_queue<QueueElement, std::vector<QueueElement>,
                      PrioritizeQueueElements>
      open_set_;

  std::vector<SearchState*> solution_;  // allocated when solve is called
  std::vector<SearchState*> explored_states_;

  enum {
    NOT_FOUND,
    FOUND
  } a_star_search_state_;  // keeps if a solution exists after solve is called
};

}  // namespace bewego
