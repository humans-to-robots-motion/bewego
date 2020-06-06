// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "astar.h"

namespace bewego {
/*!
 @ingroup GRID

* \brief Base class for 2D grid based algorithms
*
* Deriving the Grid class and the Cell class permits to generates
* easier grid algorithm. The function createNewCell is virtual just reimplement
* this function in the new class as well as the constructors which allready
* call the base one.
*/
class TwoDGrid;

/**
 * @ingroup CPP_API
 * @defgroup GRID Grid over the WS
 */

/**
  @ingroup GRID
  */
class TwoDCell {
 public:
  TwoDCell();
  TwoDCell(int i, const Eigen::Vector2d& corner, TwoDGrid* grid);
  virtual ~TwoDCell();

  Eigen::Vector2d Center() const;
  Eigen::Vector2d corner() const { return corner_; }
  Eigen::Vector2d RandomPoint() const;
  Eigen::Vector2d cell_size() const;
  int index() const { return index_; }

  bool operator==(TwoDCell c) {
    return ((c.index_) == (this->index_));
  }

 protected:
  uint32_t index_;
  Eigen::Vector2d corner_;
  TwoDGrid* grid_;
};

class TwoDGrid {
 public:
  TwoDGrid();
  TwoDGrid(const Eigen::Vector2i& num_cell, const std::vector<double>& env_size);
  TwoDGrid(double sampling_rate, const std::vector<double>& env_size);

  void setEnvSizeAndNumCell(int x, int y, std::vector<double> envSize);

  ~TwoDGrid();

  void createAllCells();

  Eigen::Vector2d cell_size() const { return cell_size_; }

  TwoDCell* getCell(const Eigen::Vector2i& cell) const;
  TwoDCell* getCell(const Eigen::Vector2d& pos) const;
  TwoDCell* getCell(uint32_t x, uint32_t y) const;
  TwoDCell* getCell(double* pos) const;
  TwoDCell* getCell(uint32_t index) const;

  bool isCellCoordInGrid(const Eigen::Vector2i& coord) const;

  Eigen::Vector2i getCellCoord(TwoDCell* ptrCell) const;
  uint32_t getNumberOfCells() const;
  TwoDCell* getNeighbour(const Eigen::Vector2i& pos, uint32_t i) const;
  Eigen::Vector2d getCoordinates(TwoDCell* cell) const;

 protected:
  virtual TwoDCell* createNewCell(
    uint32_t index,
    uint32_t x,
    uint32_t y);
  Eigen::Vector2d computeCellCorner(uint32_t x, uint32_t y);

  std::vector<TwoDCell*> cells_;
  Eigen::Vector2d origin_corner_;
  Eigen::Vector2d cell_size_;
  uint32_t nb_cells_x_;
  uint32_t nb_cells_y_;
};

class PlanGrid : public TwoDGrid {
 public:
  PlanGrid(double pace, std::vector<double> envSize, bool print_cost = true);
  void reset();
  std::pair<double, double> getMinMaxCost();
  void setCostBounds(double min, double max) {
    use_given_bounds_ = true;
    min_cost_ = min;
    max_cost_ = max;
  }

 protected:
  TwoDCell* createNewCell(uint32_t index, uint32_t x, uint32_t y);
  bool print_cost_;
  bool use_given_bounds_;
  double min_cost_;
  double max_cost_;
};

class PlanCell : public TwoDCell {
 public:
  PlanCell(int i, Eigen::Vector2i coord, Eigen::Vector2d corner,
           PlanGrid* grid);

  ~PlanCell() {}

  Eigen::Vector2i getCoord() const { return coord_; }

  double
  getCost();
  void resetCost() { cost_is_computed_ = false; }
  void setCost(double cost) {
    cost_is_computed_ = true;
    cost_ = cost;
  }

  bool getOpen() { return open_; }
  void setOpen() { open_ = true; }
  bool getClosed() { return closed_; }
  void setClosed() { closed_ = true; }
  void resetExplorationStatus() {
    open_ = false;
    closed_ = false;
  }

  bool isValid();
  void resetIsValid() { is_cell_tested_ = false; }

 private:
  Eigen::Vector2i coord_;

  bool open_;
  bool closed_;

  bool cost_is_computed_;
  double cost_;

  bool is_cell_tested_;
  bool is_valid_;
};

class PlanState : public SearchState {
 public:
  PlanState() {}
  PlanState(Eigen::Vector2i cell, PlanGrid* grid);
  PlanState(PlanCell* cell, PlanGrid* grid);

  std::vector<SearchState*> Successors(SearchState* s);

  // leaf control for an admissible heuristic function; the test of h==0
  bool leaf();
  bool equal(SearchState* other);
  bool valid();

  void setClosed(std::vector<PlanState*>& closedStates,
                 std::vector<PlanState*>& openStates);
  bool isColsed(std::vector<PlanState*>& closedStates);

  void setOpen(std::vector<PlanState*>& openStates);
  bool isOpen(std::vector<PlanState*>& openStates);

  void reset();
  void print();

  PlanCell* getCell() { return cell_; }

 protected:
  double computeLength(SearchState* parent);  // g
  double computeHeuristic(SearchState* parent = NULL,
                          SearchState* goal = NULL);  // h

 private:
  PlanGrid* grid_;
  PlanCell* cell_;
  PlanCell* previous_cell_;
};

class AStarPlanner {
 public:
  AStarPlanner();
  ~AStarPlanner();

  //! change implementation here
  unsigned int run() { return 0; }
  unsigned int init();

  bool Solve(PlanState* start, PlanState* goal);
  void reset();
  double pathCost();
  void set_pace(double pace) { pace_ = pace; }
  void set_env_size(double pace) { pace_ = pace; }

  PlanGrid* getGrid() { return grid_; }

 private:
  bool computeAStarIn2DGrid(Eigen::Vector2d source, Eigen::Vector2d target);

  std::vector<double> env_size_;
  double max_radius_;
  double pace_;
  PlanGrid* grid_;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >
      path_;
  std::vector<TwoDCell*> cell_path_;
};

}  // namespace bewego
