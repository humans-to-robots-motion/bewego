// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/derivatives/computational_graph.h>
#include <bewego/motion/objective.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>

using namespace bewego;
using namespace bewego::computational_graph;
using std::cout;
using std::endl;

std::shared_ptr<DifferentiableMap> f;
static const uint32_t NB_TESTS = 10;
static const unsigned int SEED = 0;
static const uint32_t T = 10;

TEST(computational_graph, motion_objective) {
  std::srand(SEED);
  double precision = 1e-6;
  uint32_t n = 1;
  double dt = .01;
  auto objective = std::make_shared<MotionObjective>(T, dt, n);
  objective->AddSmoothnessTerms(2, .1);
  auto network = objective->function_network();

  auto graph = std::make_shared<Graph>();
  graph->BuildFromNetwork(network);

  // Nodes -----------------------------
  // 1 : CliquesFunctionNetwork
  // T : SumMap
  // T : Scale
  // T : SquaredNormAcceleration
  // T : SquaredNorm
  // T : FiniteDifferencesAcceleration

  cout << " -- Nb of cliques : " << network->nb_cliques() << endl;
  cout << " -- Nb of nodes : " << graph->nodes().size() << endl;
  cout << " -- Nb of edges : " << graph->edges().size() << endl;

  ASSERT_EQ(graph->nodes().size(), 5 * T + 1);
  ASSERT_EQ(graph->edges().size(), 5 * T);
}