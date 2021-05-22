// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/util/misc.h>
#include <bewego/workspace/pixelmap.h>
#include <bewego/workspace/spline_grid.h>
#include <gtest/gtest.h>

using namespace bewego;
using std::cout;
using std::endl;

const double error_tolerance = 1.e-9;
const double g_error_tolerance = 1.e-2;

std::shared_ptr<LinearMap> SetUpLinearFunction() {
  Eigen::MatrixXd a(1, 2);
  a << 1, 2;
  return std::make_shared<LinearMap>(a);
}

class AnalyticalGridFunc : public ::testing::Test {
 public:
  void CheckPixelMapMatrix() {
    double resolution = .1;
    extent_t extends(0, 1, 0, 1);

    cout << "test 1" << endl;
    Eigen::MatrixXd values = Eigen::MatrixXd::Random(10, 10);
    auto pixelmap = std::make_shared<PixelMap>(resolution, extends);
    pixelmap->InitializeFromMatrix(values);
    Eigen::MatrixXd costmap = pixelmap->GetMatrix();

    // cout << "values : " << endl << values << endl;
    // cout << "costmap : " << endl << costmap << endl;
    EXPECT_EQ(values.rows(), costmap.rows());
    EXPECT_EQ(values.cols(), costmap.cols());
    EXPECT_LT((values - costmap).norm(), 1.e-12);

    cout << "test 2" << endl;
    const uint32_t buffer = 10;
    std::shared_ptr<RegressedPixelGridSpline> regressed_grid =
        InitializeRegressedPixelGridFromMatrix(resolution, values, buffer);
    costmap = regressed_grid->analytical_grid()->GetMatrix(buffer);

    // cout << "values : " << endl << values << endl;
    // cout << "costmap : " << endl << costmap << endl;
    EXPECT_EQ(values.rows(), costmap.rows());
    EXPECT_EQ(values.cols(), costmap.cols());
    EXPECT_GE(1.e-12, (values - costmap).norm());
  }

  void CheckPixelGrid(const extent_t &extends, bool origin_center_cell) {
    cout << __PRETTY_FUNCTION__
         << " , origin cell center : " << origin_center_cell << endl;

    // HERE you can set a seed
    // std::srand(1);

    Eigen::Vector2i loc;
    auto grid =
        std::make_shared<PixelMap>(resolution_, extends, origin_center_cell);
    for (loc.x() = 0; loc.x() < grid->num_cells_x(); loc.x()++) {
      for (loc.y() = 0; loc.y() < grid->num_cells_y(); loc.y()++) {
        grid->GetCell(loc) = util::Rand();
      }
    }

    // Check that the value at the center of the grid is correct
    for (loc.x() = 0; loc.x() < grid->num_cells_x(); loc.x()++) {
      for (loc.y() = 0; loc.y() < grid->num_cells_y(); loc.y()++) {
        // Get test point
        Eigen::Vector2d point;
        grid->GridToWorld(loc, point);
        bool found = grid->WorldToGrid(point, loc);
        if (!found) {
          cout << "grid->num_cells_x() : " << grid->num_cells_x() << endl;
          cout << "grid->num_cells_y() : " << grid->num_cells_y() << endl;
          cout << "origin center cell :  " << origin_center_cell << endl;
          cout << "x   : " << point.transpose() << endl;
          cout << "loc : " << loc.transpose() << endl;
        }
        EXPECT_EQ(found, true);

        // Compute and check errors
        double value_0 = grid->GetCell(loc);
        double value_1 = (*grid)(point);
        EXPECT_EQ(value_0, value_1);
      }
    }
  }

  void Check2DSaveToFile(const extent_t &extends) {
    cout << __PRETTY_FUNCTION__ << endl;

    EXPECT_EQ(extends.ExtendX(), extends.ExtendY());
    double r = (extends.ExtendX()) / 100;  // resolution

    Eigen::VectorXd a(2);
    a << 1, 2;
    double b = -.1;

    auto f1 = std::make_shared<const AffineMap>(a, b);

    auto grid = Discretize2DFunction(r, extends, f1);
    cout << "grid->num_cells_x() : " << grid->num_cells_x() << endl;
    cout << "grid->num_cells_y() : " << grid->num_cells_y() << endl;
    std::string filename = "pixel.csv";

    EXPECT_TRUE(SavePixelMapToFile(*grid, filename));

    auto f2 = LoadRegressedPixelGridFromFile(r, filename, false, 0);
    if (!Check2DGridsEqual(*grid, *f2->analytical_grid(), 1.e-6)) {
      cout << "Error: grids are not equal !!" << endl;
    }

    Eigen::Vector2i loc;

    // Check that the value at the center of the grid is correct
    // Only checks the values inside the padded area
    // The outer values are correct but not the gradients
    const int padding = 9;
    for (loc.x() = padding; loc.x() < grid->num_cells_x() - padding;
         loc.x()++) {
      for (loc.y() = padding; loc.y() < grid->num_cells_y() - padding;
           loc.y()++) {
        // Get test point
        Eigen::Vector2d point;
        grid->GridToWorld(loc, point);

        Eigen::VectorXd g1, g2;
        // Compute and check errors
        double e1, e2;
        double value_0 = grid->GetCell(loc);      // Grid point
        double value_1 = f1->ForwardFunc(point);  // Analytic function
        double value_2 = f2->ForwardFunc(point);  // Regressed function
        g1 = f1->Gradient(point);
        g2 = f2->Gradient(point);
        e1 = std::fabs(value_0 - value_1);
        e2 = std::fabs(value_0 - value_2);
        EXPECT_GE(error_tolerance, e1);
        EXPECT_GE(error_tolerance, e2);
        EXPECT_GE(g_error_tolerance, (g1 - g2).norm());
      }
    }
  }

  void Check2DAnalyticalGrid(const extent_t &extends) {
    cout << __PRETTY_FUNCTION__ << endl;

    // HERE you can set a seed
    // std::srand(1);

    Eigen::Vector2i loc;

    auto f = SetUpLinearFunction();

    // Fill grid with random values, TODO: use an analytical function
    auto grid = std::make_shared<PixelMap>(resolution_, extends);
    for (loc.x() = 0; loc.x() < grid->num_cells_x(); loc.x()++) {
      for (loc.y() = 0; loc.y() < grid->num_cells_y(); loc.y()++) {
        // grid->GetCell(loc) = util::Rand();
        Eigen::Vector2d point;
        grid->GridToWorld(loc, point);
        grid->GetCell(loc) = (*f)(point)[0];
      }
    }

    if (verbose_) {
      cout << " - origin      : " << grid->origin().transpose() << endl;
      cout << " - resolution  : " << grid->resolution() << endl;
      cout << " - num_cells_x : " << grid->num_cells_x() << endl;
      cout << " - num_cells_y : " << grid->num_cells_y() << endl;
      cout << "Matrix : " << endl << grid->GetMatrix() << endl;
      cout << "Construct splined grids" << endl;
    }

    // Construct interpolation grids
    // 1) grid constructed by setting the values manualy
    auto s_grid_1 =
        std::make_shared<AnalyticPixelMapSpline>(resolution_, extends);
    for (loc.x() = 0; loc.x() < grid->num_cells_x(); loc.x()++) {
      for (loc.y() = 0; loc.y() < grid->num_cells_y(); loc.y()++) {
        s_grid_1->GetCell(loc) = grid->GetCell(loc);
      }
    }
    s_grid_1->InitializeSplines();

    // 2) grid constructed using the copy constructor
    auto s_grid_2 = std::make_shared<AnalyticPixelMapSpline>(*grid);
    s_grid_2->InitializeSplines();

    // Check that the value at the center of the grid is correct
    const int padding = 1;  // Only checks the values inside the padded area
    for (loc.x() = padding; loc.x() < grid->num_cells_x() - padding;
         loc.x()++) {
      for (loc.y() = padding; loc.y() < grid->num_cells_y() - padding;
           loc.y()++) {
        double value_0 = grid->GetCell(loc);
        double value_1 = s_grid_1->GetCell(loc);
        double value_2 = s_grid_2->GetCell(loc);
        ASSERT_EQ(value_0, value_1);
        ASSERT_EQ(value_0, value_2);

        // Get test point
        Eigen::Vector2d point;
        grid->GridToWorld(loc, point);

        // Compute and check errors
        double e1, e2;
        value_1 = s_grid_1->CalculateSplineValue(point);
        value_2 = s_grid_2->CalculateSplineValue(point);

        if (verbose_) {
          cout << "loc     : " << loc.transpose() << endl;
          cout << "value_0 : " << value_0 << endl;
          cout << "value_1 : " << value_1 << endl;
          cout << "value_2 : " << value_2 << endl;
        }

        e1 = std::fabs(value_0 - value_1);
        e2 = std::fabs(value_0 - value_2);

        double v_tol = error_tolerance;
        ASSERT_GE(v_tol, e1);
        ASSERT_GE(v_tol, e2);

        Eigen::Vector2d g_1, g_2;

        // Same but with gradient
        double g_tol = error_tolerance;
        g_1 = s_grid_1->CalculateSplineGradient(point);
        g_2 = s_grid_2->CalculateSplineGradient(point);
        ASSERT_NEAR(g_1.x(), g_2.x(), g_tol);
        ASSERT_NEAR(g_1.y(), g_2.y(), g_tol);
      }
    }
  }

  void Check2DAnalyticalGridFunction(const extent_t &extends) {
    cout << __PRETTY_FUNCTION__ << endl;
    double r = resolution_;
    Eigen::VectorXd a(2);
    a << 1, 2;
    double b = -.1;
    auto f1 = std::make_shared<const AffineMap>(a, b);
    auto f2 = RegressedGridFrom2DFunction(r, extends, f1);
    auto grid = Discretize2DFunction(r, extends, f1);
    cout << "grid->num_cells_x() : " << grid->num_cells_x() << endl;
    cout << "grid->num_cells_y() : " << grid->num_cells_y() << endl;

    Eigen::Vector2i loc;

    // Checks that the value at the center of the grid is correct
    // Only checks the values inside the padded area
    // The outer values are correct but not the gradients
    const int padding = 9;
    for (loc.x() = padding; loc.x() < grid->num_cells_x() - padding;
         loc.x()++) {
      for (loc.y() = padding; loc.y() < grid->num_cells_y() - padding;
           loc.y()++) {
        // Get test point
        Eigen::Vector2d point;
        grid->GridToWorld(loc, point);

        Eigen::VectorXd g1, g2;
        // Compute and check errors
        double e1, e2;
        double value_0 = grid->GetCell(loc);      // Grid point
        double value_1 = f1->ForwardFunc(point);  // Analytic function
        double value_2 = f2->ForwardFunc(point);  // Regressed function
        g1 = f1->Gradient(point);
        g2 = f2->Gradient(point);
        if (verbose_) {
          cout << "g1 : " << g1.transpose() << endl;
          cout << "g2 : " << g2.transpose() << endl;
          cout << "g3 : "
               << DifferentiableMap::FiniteDifferenceJacobian(*f2, point)
               << endl;
        }
        e1 = std::fabs(value_0 - value_1);
        e2 = std::fabs(value_0 - value_2);
        ASSERT_GE(error_tolerance, e1);
        ASSERT_GE(error_tolerance, e2);
        ASSERT_GE(g_error_tolerance, (g1 - g2).norm());
      }
    }
  }

  void SetUp() { verbose_ = false; }

 protected:
  bool verbose_;
  double resolution_;
};

TEST_F(AnalyticalGridFunc, CheckPixelGrid) {
  verbose_ = false;
  CheckPixelMapMatrix();

  resolution_ = 0.01;

  std::vector<extent_t> sizes;
  sizes.push_back(extent_t(0, 2, 0, 2));
  sizes.push_back(extent_t(-1, 1, -1, 1));
  sizes.push_back(extent_t(-5, -1, -3, 1));

  for (auto s : sizes) {
    CheckPixelGrid(s, false);
  }
  for (auto s : sizes) {
    CheckPixelGrid(s, true);
  }
}

TEST_F(AnalyticalGridFunc, Check2DAnalyticalGrid) {
  verbose_ = false;

  resolution_ = 0.01;

  std::vector<extent_t> sizes;
  sizes.push_back(extent_t(0, 2, 0, 2));
  sizes.push_back(extent_t(-1, 1, -1, 1));
  sizes.push_back(extent_t(-5, -1, -3, 1));

  for (auto s : sizes) {
    Check2DAnalyticalGrid(s);
  }

  /*  TODO fix that, same as bellow with gradient...
  for (int i = 0; i < 1; i++) {
    // Only the ones that have (0, 0) as the origin
    Check2DSaveToFile(sizes[i]);
  }
  */
}

TEST_F(AnalyticalGridFunc, Check2DAllGrids) {
  verbose_ = false;

  resolution_ = 1;  // TODO change that

  std::vector<extent_t> sizes;
  sizes.push_back(extent_t(0, 30, 0, 30));
  // sizes.push_back(extent_t(0, 2, 0, 2));
  // sizes.push_back(extent_t(-1, 1, -1, 1));
  // sizes.push_back(extent_t(-5, -1, -3, 1));

  for (auto s : sizes) {
    Check2DAnalyticalGridFunction(s);
  }
}

TEST(AnalyticalGrid, CheckCrown) {
  bool verbose = false;
  extent_t extend(0, 1, 0, 1);
  auto pixelmap = std::make_shared<PixelMap>(.1, extend);
  uint32_t n = pixelmap->num_cells_x();
  cout << "Nb of cells : " << n << endl;
  pixelmap->InitializeFromMatrix(Eigen::MatrixXd::Zero(n, n));
  Eigen::MatrixXd M;

  cout << "Set crown per crown..." << endl;
  pixelmap->SetCrown(1, 0);
  pixelmap->SetCrown(2, 1);
  pixelmap->SetCrown(3, 2);
  cout << "Get Matrix..." << endl;
  M = pixelmap->GetMatrix();
  ASSERT_TRUE(M.rows() == n);
  ASSERT_TRUE(M.cols() == n);
  if (verbose) {
    cout << "M : " << endl << M << endl;
  }
  Eigen::MatrixXd A(10, 10);
  A.row(0) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
  A.row(1) << 1, 2, 2, 2, 2, 2, 2, 2, 2, 1;
  A.row(2) << 1, 2, 3, 3, 3, 3, 3, 3, 2, 1;
  A.row(3) << 1, 2, 3, 0, 0, 0, 0, 3, 2, 1;
  A.row(4) << 1, 2, 3, 0, 0, 0, 0, 3, 2, 1;
  A.row(5) << 1, 2, 3, 0, 0, 0, 0, 3, 2, 1;
  A.row(6) << 1, 2, 3, 0, 0, 0, 0, 3, 2, 1;
  A.row(7) << 1, 2, 3, 3, 3, 3, 3, 3, 2, 1;
  A.row(8) << 1, 2, 2, 2, 2, 2, 2, 2, 2, 1;
  A.row(9) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
  EXPECT_NEAR((A - M).cwiseAbs().maxCoeff(), 0., 1e-6);

  cout << "Set all crowns..." << endl;
  pixelmap->SetBorderCrown(9, 3);
  M = pixelmap->GetMatrix();
  ASSERT_TRUE(M.rows() == n);
  ASSERT_TRUE(M.cols() == n);
  A.row(0) << 9, 9, 9, 9, 9, 9, 9, 9, 9, 9;
  A.row(1) << 9, 9, 9, 9, 9, 9, 9, 9, 9, 9;
  A.row(2) << 9, 9, 9, 9, 9, 9, 9, 9, 9, 9;
  A.row(3) << 9, 9, 9, 0, 0, 0, 0, 9, 9, 9;
  A.row(4) << 9, 9, 9, 0, 0, 0, 0, 9, 9, 9;
  A.row(5) << 9, 9, 9, 0, 0, 0, 0, 9, 9, 9;
  A.row(6) << 9, 9, 9, 0, 0, 0, 0, 9, 9, 9;
  A.row(7) << 9, 9, 9, 9, 9, 9, 9, 9, 9, 9;
  A.row(8) << 9, 9, 9, 9, 9, 9, 9, 9, 9, 9;
  A.row(9) << 9, 9, 9, 9, 9, 9, 9, 9, 9, 9;
  EXPECT_NEAR((A - M).cwiseAbs().maxCoeff(), 0., 1e-6);
}
