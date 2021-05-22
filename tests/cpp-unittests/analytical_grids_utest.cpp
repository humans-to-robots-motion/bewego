// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Nathan Ratliff, Jim Mainprice
#include <bewego/derivatives/atomic_operators.h>
#include <bewego/workspace/analytical_grid.h>
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

using namespace bewego;
using std::cout;
using std::endl;

double LwrWeight(double square_distance) { return exp(-.5 * square_distance); }

double LwrWeight(const Eigen::VectorXd& x, const Eigen::VectorXd& x_data,
                 const Eigen::MatrixXd& D) {
  return LwrWeight(MahalanobisSquareDistance(x, x_data, D));
}

TEST(MahalanobisToolsTest, BasicSquareDistance) {
  bool verbose = false;

  Eigen::MatrixXd D = Eigen::VectorXd::Ones(3).asDiagonal();
  Eigen::VectorXd x1 = Eigen::VectorXd::Random(3);
  Eigen::VectorXd x2 = Eigen::VectorXd::Random(3);

  double expected_square_distance = pow((x1 - x2).norm(), 2);
  double calculated_square_distance = MahalanobisSquareDistance(x1, x2, D);

  if (verbose) {
    cout << expected_square_distance << endl;
    cout << calculated_square_distance << endl;
  }
  EXPECT_NEAR(expected_square_distance, calculated_square_distance, 1e-7);
}

Eigen::MatrixXd RandomMahalanobisMatrix(uint32_t dim) {
  Eigen::VectorXd d = Eigen::VectorXd::Random(3).array().abs().matrix();
  return d.asDiagonal();
}

TEST(MahalanobisToolsTest, TestNeighborhoodDistanceThreshold) {
  bool verbose = false;
  Eigen::MatrixXd D = RandomMahalanobisMatrix(3);

  Eigen::VectorXd x_data = Eigen::VectorXd::Random(3);
  Eigen::VectorXd x = Eigen::VectorXd::Random(3);

  double weight_at_x = LwrWeight(x, x_data, D);
  double equiv_threshold_dist = DistanceAtWeightThreshold(weight_at_x);
  if (verbose) {
    cout << equiv_threshold_dist << endl;
    cout << MahalanobisDistance(x, x_data, D) << endl;
  }
  EXPECT_NEAR(equiv_threshold_dist, MahalanobisDistance(x, x_data, D), 1e-7);
}

TEST(MahalanobisToolsTest, TestRescaleMahalanobisMetric) {
  bool verbose = false;
  Eigen::MatrixXd D = RandomMahalanobisMatrix(3);
  Eigen::VectorXd x_data = Eigen::VectorXd::Random(3);
  Eigen::VectorXd x = Eigen::VectorXd::Random(3);
  double initial_distance = MahalanobisDistance(x, x_data, D);
  double desired_weight = .001;

  if (verbose) cout << D << endl;

  RescaleMahalanobisMetric(initial_distance, desired_weight, &D);

  if (verbose) cout << D << endl;
  double scaled_weight = LwrWeight(x, x_data, D);
  if (verbose) {
    cout << desired_weight << endl;
    cout << scaled_weight << endl;
  }
}

class FriendlyAnalyticalGrid : public AnalyticalGrid {
 public:
  FriendlyAnalyticalGrid() {}
  FriendlyAnalyticalGrid(double cell_size, const std::vector<double>& env_box,
                         bool corner)
      : AnalyticalGrid(cell_size, env_box, corner) {}
  FriendlyAnalyticalGrid(double cell_size, const std::vector<double>& env_box,
                         const Eigen::MatrixXd& mahalanobis_metric,
                         double neighborhood_threshold, bool corner)
      : AnalyticalGrid(cell_size, env_box, mahalanobis_metric,
                       neighborhood_threshold, corner) {}

  FRIEND_TEST(AnalyticalGridTest, ComputeNeighbors);
};

class AnalyticalGridTest : public ::testing::Test {
 public:
  void SetUp() {
    analytical_grid_ = AnalyticalGrid(
        .02, CreateEnvironmentBox(-.1, .1, -.1, .1, -.1, .1), false);
    Eigen::VectorXd a(3);
    a << 1, 2, 3;
    double b = -.1;
    affine_function_ = std::make_shared<AffineMap>(a, b);
  }

  void InitializeValues() {
    for (uint32_t i = 0; i < analytical_grid_.getNumCells(DIM_X); ++i) {
      for (uint32_t j = 0; j < analytical_grid_.getNumCells(DIM_Y); ++j) {
        for (uint32_t k = 0; k < analytical_grid_.getNumCells(DIM_Z); ++k) {
          Eigen::Vector3i grid_cell(i, j, k);
          Eigen::Vector3d x;
          analytical_grid_.gridToWorld(grid_cell, x);
          analytical_grid_.setCell(grid_cell, (*affine_function_)(x)[0]);
        }
      }
    }
  }

  void InitializeValuesForCenterDistance() {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (uint32_t i = 0; i < analytical_grid_.getNumCells(DIM_X); ++i) {
      for (uint32_t j = 0; j < analytical_grid_.getNumCells(DIM_Y); ++j) {
        for (uint32_t k = 0; k < analytical_grid_.getNumCells(DIM_Z); ++k) {
          Eigen::Vector3i grid_cell(i, j, k);
          Eigen::Vector3d x;
          analytical_grid_.gridToWorld(grid_cell, x);
          analytical_grid_.setCell(grid_cell, (x - center).norm());
        }
      }
    }
  }

  void InitializeMahalanobisMetric() {
    Eigen::MatrixXd mahalanobis_metric = Eigen::VectorXd::Ones(3).asDiagonal();
    neighborhood_threshold_ = .05;
    weight_threshold_ = .01;
    analytical_grid_.SetMahalanobisDistanceMetric(
        mahalanobis_metric, neighborhood_threshold_, weight_threshold_);
  }

  void InitializeAll() {
    InitializeMahalanobisMetric();
    InitializeValues();
  }

  void ValidateGridPoint(const Eigen::Vector3i& query_cell, double precision,
                         bool verbose) {
    Eigen::Vector3d query_pt = analytical_grid_.gridToWorld(query_cell);
    double potential_value = analytical_grid_.CalculatePotential(query_pt);
    double actual_value = (*affine_function_)(query_pt)[0];
    if (verbose) {
      cout << "------------------------" << endl;
      cout << "query_cell: " << query_cell.transpose()
           << ", query_pt: " << query_pt.transpose() << endl;
      cout << actual_value << endl;
      cout << potential_value << endl;
    }
    EXPECT_NEAR(actual_value, potential_value, precision);
  }

 protected:
  double neighborhood_threshold_;
  double weight_threshold_;
  AnalyticalGrid analytical_grid_;
  std::shared_ptr<AffineMap> affine_function_;
};

std::vector<Eigen::Vector3i> BruteForceNeighbors(double distance_threshold,
                                                 double resolution) {
  std::vector<Eigen::Vector3i> neighbors;
  double square_distance_threshold = distance_threshold * distance_threshold;
  int cell_radius = ceil(distance_threshold / resolution);

  for (int i_offset = -cell_radius; i_offset <= cell_radius; ++i_offset) {
    for (int j_offset = -cell_radius; j_offset <= cell_radius; ++j_offset) {
      for (int k_offset = -cell_radius; k_offset <= cell_radius; ++k_offset) {
        Eigen::VectorXd point(3);
        point << resolution * i_offset, resolution * j_offset,
            resolution * k_offset;
        double square_distance = point.dot(point);
        if (square_distance <= square_distance_threshold) {
          neighbors.push_back(Eigen::Vector3i(i_offset, j_offset, k_offset));
        }
      }
    }
  }
  return neighbors;
}

// Tests agains a brute force calculation specifically for the Euclidean metrix.
TEST_F(AnalyticalGridTest, ComputeNeighbors2) {
  bool verbose = false;
  InitializeMahalanobisMetric();
  const std::vector<Eigen::Vector3i>& neighbors = analytical_grid_.neighbors();
  std::vector<Eigen::Vector3i> brute_neighbors = BruteForceNeighbors(
      neighborhood_threshold_, analytical_grid_.getResolution());

  if (verbose) {
    cout << neighbors.size() << endl;
    cout << brute_neighbors.size() << endl;
  }

  EXPECT_EQ(brute_neighbors.size(), neighbors.size());
  for (uint32_t i = 0; i < neighbors.size(); ++i) {
    // Each of these neighbors should be found in brute neighbors.
    EXPECT_FALSE(std::find(brute_neighbors.begin(), brute_neighbors.end(),
                           neighbors[i]) == brute_neighbors.end());
  }
}

TEST_F(AnalyticalGridTest, FillSpace) {
  InitializeAll();
  bool verbose = false;
  if (verbose) {
    for (uint32_t i = 0; i < analytical_grid_.getNumCells(DIM_X); ++i) {
      for (uint32_t j = 0; j < analytical_grid_.getNumCells(DIM_Y); ++j) {
        for (uint32_t k = 0; k < analytical_grid_.getNumCells(DIM_Z); ++k) {
          Eigen::Vector3i grid_cell(i, j, k);
          double value = analytical_grid_.getCell(grid_cell);
          Eigen::Vector3d x;
          analytical_grid_.gridToWorld(grid_cell, x);
          cout << "Cell at (" << i << "," << j << "," << k << ") --> (" << x.x()
               << "," << x.y() << "," << x.z() << ") = " << value << endl;
        }
      }
    }
  }

  // Validate the single center point.
  Eigen::Vector3i query_cell(analytical_grid_.getNumCells(DIM_X) / 2,
                             analytical_grid_.getNumCells(DIM_Y) / 2,
                             analytical_grid_.getNumCells(DIM_Z) / 2);
  ValidateGridPoint(query_cell, 1e-7, verbose);
}

TEST_F(AnalyticalGridTest, ValidateInteriorPoints) {
  bool verbose = false;
  InitializeAll();
  int buffer = 2;
  for (uint32_t i = buffer; i < analytical_grid_.getNumCells(DIM_X) - buffer;
       ++i) {
    for (uint32_t j = buffer; j < analytical_grid_.getNumCells(DIM_Y) - buffer;
         ++j) {
      for (uint32_t k = buffer;
           k < analytical_grid_.getNumCells(DIM_Z) - buffer; ++k) {
        Eigen::Vector3i query_cell(i, j, k);
        ValidateGridPoint(query_cell, 1e-7, verbose);
      }
    }
  }
}

TEST_F(AnalyticalGridTest, ValidateAllRelevantPoints) {
  bool verbose = false;
  InitializeAll();

  int buffer = 0;
  for (int32_t i = -buffer; i < analytical_grid_.getNumCells(DIM_X) + buffer;
       ++i) {
    for (int32_t j = -buffer; j < analytical_grid_.getNumCells(DIM_Y) + buffer;
         ++j) {
      for (int32_t k = -buffer;
           k < analytical_grid_.getNumCells(DIM_Z) + buffer; ++k) {
        Eigen::Vector3i query_cell(i, j, k);
        ValidateGridPoint(query_cell, 1e-2, verbose);
      }
    }
  }
}

TEST_F(AnalyticalGridTest, ValidateDegredationToZero) {
  bool verbose = false;
  InitializeAll();

  Eigen::Vector3d world_pt =
      analytical_grid_.gridToWorld(Eigen::Vector3i::Zero());
  Eigen::Vector3d increment = -.5 / 1000. * Eigen::Vector3d::Ones();

  double prev_diff = 0;
  double prev_value = 0;
  bool switch_detected = false;
  for (int i = 0; i < 100; ++i) {
    world_pt += increment;

    double analytical_value = (*affine_function_)(world_pt)[0];
    double value = analytical_grid_.CalculatePotential(world_pt);
    double new_diff = value - analytical_value;
    EXPECT_GT(new_diff, prev_diff);

    if (verbose) {
      cout << "------------------------" << endl;
      cout << value << endl;
      cout << analytical_value << endl;
      cout << new_diff << endl;  // Should be positive.
    }

    double value_diff = fabs(prev_value) - fabs(value);
    switch_detected = switch_detected || value_diff > 0;
    if (switch_detected) {
      // For the remainder of the indices, the potential values should be
      // approaching 0.
      EXPECT_GE(value_diff, 0);
    } else {
      EXPECT_LT(i, 39);  // The inflection point currently happens at index 39.
    }

    prev_diff = new_diff;
    prev_value = value;
  }
}

TEST_F(AnalyticalGridTest, RegressedVoxelGrid) {
  bool verbose = false;
  InitializeAll();

  RegressedVoxelGrid f(analytical_grid_);

  for (uint32_t i = 0; i < 10; ++i) {
    Eigen::Vector3d x = Eigen::Vector3d::Random() / 10;
    double value = f(x)[0];
    if (verbose) cout << value << endl;
    EXPECT_EQ(f(x)[0], analytical_grid_.CalculatePotential(x));
  }
}

TEST_F(AnalyticalGridTest, ClosestDistanceFromGradient) {
  bool verbose = false;
  InitializeMahalanobisMetric();
  InitializeValuesForCenterDistance();

  SquaredRegressedVoxelGrid f(analytical_grid_);

  Eigen::Vector3d x1(.05, .02, .06);
  Eigen::Vector3d x2(-.025, .02, .06);
  double value1 = f(x1)[0];
  double value2 = f(x2)[0];
  Eigen::VectorXd g1 = f.Gradient(x1);
  Eigen::VectorXd g2 = f.Gradient(x2);

  Eigen::Vector3d diff1 = x1 - g1;
  Eigen::Vector3d diff2 = x2 - g2;

  if (verbose) {
    cout << "------------------------" << endl;
    cout << x1.transpose() << endl;
    cout << value1 << endl;
    cout << -g1.transpose() << endl;
    cout << diff1.transpose() << endl;
  }
  EXPECT_NEAR(diff1.x(), 0,
              .1);  // Precision on x is much worse: close to border.
  EXPECT_NEAR(diff1.y(), 0, 1e-3);
  EXPECT_NEAR(diff1.z(), 0, 1e-3);
  if (verbose) {
    cout << "------------------------" << endl;
    cout << x2.transpose() << endl;
    cout << value2 << endl;
    cout << -g2.transpose() << endl;
    cout << diff2.transpose() << endl;
  }
  EXPECT_NEAR(diff2.x(), 0, 2e-3);
  EXPECT_NEAR(diff2.y(), 0, 2e-3);
  EXPECT_NEAR(diff2.z(), 0, 2e-3);
}

TEST_F(AnalyticalGridTest, ClosestDistanceFromGradientUsingSpecialFunction) {
  bool verbose = false;
  InitializeMahalanobisMetric();
  InitializeValuesForCenterDistance();

  SquaredRegressedVoxelGrid f(analytical_grid_);

  Eigen::Vector3d x1(.05, .02, .06);
  Eigen::Vector3d x2(-.025, .02, .06);
  Eigen::Vector3d ng1 = f.CalculateNegativeGradient(x1);
  Eigen::Vector3d ng2 = f.CalculateNegativeGradient(x2);

  Eigen::Vector3d diff1 = x1 + ng1;
  Eigen::Vector3d diff2 = x2 + ng2;

  if (verbose) {
    cout << "------------------------" << endl;
    cout << x1.transpose() << endl;
    cout << ng1.transpose() << endl;
    cout << diff1.transpose() << endl;
  }
  EXPECT_NEAR(diff1.x(), 0,
              .1);  // Precision on x is much worse: close to border.
  EXPECT_NEAR(diff1.y(), 0, 1e-3);
  EXPECT_NEAR(diff1.z(), 0, 1e-3);
  if (verbose) {
    cout << "------------------------" << endl;
    cout << x2.transpose() << endl;
    cout << ng2.transpose() << endl;
    cout << diff2.transpose() << endl;
  }
  EXPECT_NEAR(diff2.x(), 0, 2e-3);
  EXPECT_NEAR(diff2.y(), 0, 2e-3);
  EXPECT_NEAR(diff2.z(), 0, 2e-3);
}

// int main(int argc, char** argv) {
//   testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
