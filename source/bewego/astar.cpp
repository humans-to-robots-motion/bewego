// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include "astar.h"
#include "chrono.h"
#include <iostream>

using namespace std;
using namespace bewego;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
SearchState::SearchState() : _f(0), _g(0), _h(0) {}

double SearchState::computeCost(SearchState* parent, SearchState* goal) {
  g_ = Length(parent);
  h_ = Heuristic(parent, goal);
  return f = (g_ + h_);
}

bool SearchState::is_leaf() { return (0 == h_); }

bool SearchState::equal(SearchState* other) {
  cout << "equal(SearchState* other) not implemented" << endl;
  return false;
}

double SearchState::f() const { return f_; }
double SearchState::g() const { return g_; }
double SearchState::h() const { return h_; }

vector<SearchState*> SearchState::getSuccessors(SearchState* s) {
  vector<SearchState*> successors;
  cout << "SearchState::getSuccessors() Not implemented" << endl;
  return successors;
}

double SearchState::Length(SearchState* parent) { return 0; }
double SearchState::Heuristic(SearchState* parent = NULL,
                              SearchState* goal = NULL) {
  return 0;
}

bool SearchState::is_closed(vector<SearchState*>& closed_search_states) {
  bool is_closed = false;
  for (unsigned i = 0; i < closed_search_states.size(); i++) {
    if (equal(closed_search_states[i])) {
      is_closed = true;
      break;
    }
  }
  return is_closed;
}

void SearchState::set_closed(
    std::vector<SearchState*>& closed_search_states,
    std::vector<SearchState*>& open_search_states) {
  for (vector<SearchState*>::iterator it = open_search_states.begin();
       it != open_search_states.end(); ++it) {
    if ((*it)->equal(this)) {
      open_search_states.erase(it);
      break;
    }
  }
  closed_search_states.push_back(this);
}

bool SearchState::is_open(vector<SearchState*>& open_search_states) {
  bool is_open = false;
  for (unsigned i = 0; i < open_search_states.size(); i++) {
    if (this->equal(open_search_states[i])) {
      is_open = true;
      break;
    }
  }
  return is_open;
}

void SearchState::set_open(std::vector<SearchState*>& open_search_states) {
  open_search_states.push_back(this);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
/**
 * Tree Element Class
 */
TreeNode::TreeNode(SearchState* st, TreeNode* prnt = NULL) {
  if (st != NULL) {
    search_state_ = st;
    parent_ = prnt;
  } else {
    cerr << "Error in TreeNode::TreeNode(SearchState *st, TreeNode "
            "*prnt=NULL)\n";
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
/**
 * A Star
 * backwards solution
 */
vector<SearchState*> AStar::getSolution(QueueElement q_tmp) {
  a_star_search_state_ = FOUND;
  TreeNode* solution_leaf = q_tmp.getTreeNode();
  TreeNode* node = solution_leaf;
  
  while (node) {
    solution_.push_back(node->search_state());
    node = node->parent();
  }

  return solution_;
}

/**
 *
 */
bool AStar::isGoal(SearchState* state) {
  if (goal_is_defined_) {
    return goal_->equal(state);
  } else {
    return (false);
  }
}

void AStar::cleanSearchStates() {
  for (unsigned i = 0; i < explored_.size(); i++) {
    explored_[i]->reset();
  }
}

/**
 * A Star
 * solving function (main)
 */
vector<SearchState*> AStar::solve(SearchState* initial_search_state) {
  double tu, ts;
  cout << "start solve" << endl;
  ChronoOn();

  //    _Explored.reserve(20000);

  a_star_search_state_ = NOT_FOUND;
  solution_leaf_ = NULL;

  // initial_search_state->computeCost(NULL);
  root_ = new TreeNode(initial_search_state);

  vector<SearchState*> closed_set;
  vector<SearchState*> open_set;
  open_set.push_back(initial_search_state);

  open_set_.push(*new QueueElement(root_));
  explored_.push_back(initial_search_state);

  QueueElement q_tmp;

  while (!open_set.empty()) {
    q_tmp = open_set.top();
    open_set.pop();

    SearchState* current_search_state = q_tmp.tree_node()->search_state();
    // cout << "SearchState = "<< currentSearchState << endl;
    current_search_state->set_closed(open_set, closed_set);

    /* The solution is found */
    if (current_search_state->leaf() || is_goal(current_search_state)) {
      CleanSearchStates();
      ChronoPrint("");
      ChronoTimes(&tu, &ts);
      ChronoOff();
      cout << "Number of explored states = " << _Explored.size() << endl;
      return Solution(q_tmp);
    }

    TreeNode* parent = q_tmp.getTreeNode()->getParent();
    SearchState* parent_state = NULL;
    if (parent != NULL) {
      parent_state = parent->getSearchState();
    }

    vector<SearchState*> branched_states =
        currentSearchState->getSuccessors(parent_state);

    for (unsigned int i = 0; i < branched_states.size(); i++) {
      if ((branched_states[i] != NULL) &&
          (branched_states[i]->isValid())) {
        if (!((parent != NULL) && (parent->search_state()->valid()) &&
              (parent->getSearchState()->equal(branched_states[i])))) {
          if (!(branched_states[i]->is_closed(closed_set))) {
            if (!(branched_states[i]->is_open(open_set))) {
              branched_states[i]->computeCost(current_search_state, goal_);
              branched_states[i]->setOpen(openSet);

              explored_.push_back(branched_states[i]);
              open_set_.push(*new QueueElement(new TreeNode(
                  branched_states[i], (q_tmp.getTreeNode()))));
            }
          }
        }
      }
    }
  }

  CleanSearchStates();
  ChronoPrint("");
  ChronoTimes(&tu, &ts);
  ChronoOff();

  if (NOT_FOUND == a_star_search_state_) {
    cerr << "The Solution does not exist\n";
    return _Solution;
  }

  return _Solution;
}
