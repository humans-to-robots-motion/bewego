// Copyright (c) 2019, University of Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/util/misc.h>
#include <bewego/workspace/extent.h>
#include <bewego/workspace/pixelmap.h>
#include <gtest/gtest.h>

using namespace bewego;
using std::cout;
using std::endl;

const double error = 1e-05;
const double nb_test_points = 10;
const double resolution = 0.01;

TEST(ExtentBox, main) {
  ExtentBox box(0, 1, 2, 3, 4, 5);

  ASSERT_TRUE(box.extent(0, ExtentBox::Min) == 0);
  ASSERT_TRUE(box.extent(0, ExtentBox::Max) == 1);
  ASSERT_TRUE(box.extent(1, ExtentBox::Min) == 2);
  ASSERT_TRUE(box.extent(1, ExtentBox::Max) == 3);
  ASSERT_TRUE(box.extent(2, ExtentBox::Min) == 4);
  ASSERT_TRUE(box.extent(2, ExtentBox::Max) == 5);

  ASSERT_TRUE(box.extent(0, ExtentBox::Min) == box.x_min());
  ASSERT_TRUE(box.extent(0, ExtentBox::Max) == box.x_max());
  ASSERT_TRUE(box.extent(1, ExtentBox::Min) == box.y_min());
  ASSERT_TRUE(box.extent(1, ExtentBox::Max) == box.y_max());
  ASSERT_TRUE(box.extent(2, ExtentBox::Min) == box.z_min());
  ASSERT_TRUE(box.extent(2, ExtentBox::Max) == box.z_max());
}

bool CheckPixelMapMatrix() {
  double resolution = .1;
  ExtentBox extends(0, 1, 0, 1);

  Eigen::MatrixXd values = Eigen::MatrixXd::Random(10, 10);
  auto pixelmap = std::make_shared<PixelMap>(resolution, extends);
  pixelmap->InitializeFromMatrix(values);
  Eigen::MatrixXd costmap = pixelmap->GetMatrix();

  EXPECT_EQ(values.rows(), costmap.rows());
  EXPECT_EQ(values.cols(), costmap.cols());
  EXPECT_LT((values - costmap).norm(), 1.e-12);

  return true;
}

bool CheckPixelGrid(const ExtentBox& extends, bool origin_center_cell) {
  cout << __PRETTY_FUNCTION__
       << " , origin cell center : " << origin_center_cell << endl;

  // HERE you can set a seed
  // std::srand(1);

  Eigen::Vector2i loc;
  auto grid =
      std::make_shared<PixelMap>(resolution, extends, origin_center_cell);
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
  return true;
}

TEST(PixelMap, Check2DAllGrids) {
  ASSERT_TRUE(CheckPixelMapMatrix());

  std::vector<ExtentBox> sizes;
  sizes.push_back(ExtentBox(-1, 1, -1, 1));
  sizes.push_back(ExtentBox(0, 2, 0, 2));
  sizes.push_back(ExtentBox(-5, -1, -3, 1));

  for (auto s : sizes) {
    ASSERT_TRUE(CheckPixelGrid(s, false));
  }
  for (auto s : sizes) {
    ASSERT_TRUE(CheckPixelGrid(s, true));
  }
}