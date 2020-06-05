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
  TwoDCell(int i, Eigen::Vector2d corner, TwoDGrid* grid);
  virtual ~TwoDCell();

  Eigen::Vector2d getCenter();
  Eigen::Vector2d getCorner() { return _corner; }
  Eigen::Vector2d getRandomPoint();
  Eigen::Vector2d getCellSize();

  int getIndex() { return _index; }

  bool operator==(TwoDCell otherCell) {
    return ((otherCell._index) == (this->_index));
  }

 protected:
  int _index;
  Eigen::Vector2d _corner;
  TwoDGrid* _grid;
};

class TwoDGrid {
 public:
  TwoDGrid();
  TwoDGrid(Eigen::Vector2i numCell, std::vector<double> envSize);
  TwoDGrid(double samplingRate, std::vector<double> envSize);

  void setEnvSizeAndNumCell(int x, int y, std::vector<double> envSize);

  ~TwoDGrid();

  void createAllCells();

  Eigen::Vector2d getCellSize() { return _cellSize; }

  TwoDCell* getCell(const Eigen::Vector2i& cell);
  TwoDCell* getCell(int x, int y);
  TwoDCell* getCell(Eigen::Vector2d pos);
  TwoDCell* getCell(double* pos);
  TwoDCell* getCell(unsigned int index);

  bool isCellCoordInGrid(const Eigen::Vector2i& coord);

  Eigen::Vector2i getCellCoord(TwoDCell* ptrCell);
  int getNumberOfCells();
  TwoDCell* getNeighbour(const Eigen::Vector2i& pos, int i);
  Eigen::Vector2d getCoordinates(TwoDCell* cell);

 protected:
  virtual TwoDCell* createNewCell(unsigned int index, unsigned int x,
                                  unsigned int y);
  Eigen::Vector2d computeCellCorner(int x, int y);

  std::vector<TwoDCell*> _cells;
  Eigen::Vector2d _originCorner;
  Eigen::Vector2d _cellSize;

  unsigned int _nbCellsX;
  unsigned int _nbCellsY;
};


class PlanGrid : public TwoDGrid {
 public:
  PlanGrid(double pace, std::vector<double> envSize,
           bool print_cost = true);

  TwoDCell* createNewCell(unsigned int index, unsigned int x, unsigned int y);

  void reset();
  std::pair<double, double> getMinMaxCost();
  void draw();
  void setCostBounds(double min, double max) {
    use_given_bounds_ = true;
    min_cost_ = min;
    max_cost_ = max;
  }

 private:
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

  Eigen::Vector2i getCoord() { return coord_; }

  double
  getCost(); // { std::cout << " Warning not implemented"  << std::endl; }
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
  double computeLength(SearchState* parent); // g
  double computeHeuristic(SearchState* parent = NULL,
                          SearchState* goal = NULL); // h 

 private:
  PlanGrid* grid_;
  PlanCell* cell_;
  PlanCell* previous_cell_;
};

/**
class AStarPlanner {
 public:
  AStarPlanner();
  ~AStarPlanner();

  // Trajectory* computeRobotTrajectory(confPtr_t source, confPtr_t target);
  Trajectory* getSimplePath(std::vector<double> goal,
                            std::vector<std::vector<double> >& path);

  //! change implementation here
  unsigned int run() { return 0; }
  unsigned int init();

  void reset();
  void draw();
  double pathCost();
  void allow_smoothing(bool state);

  void set_pace(double pace) { pace_ = pace; }

  PlanGrid* getGrid() { return grid_; }

 private:
  bool computeAStarIn2DGrid(Eigen::Vector2d source, Eigen::Vector2d target);
  bool solveAStar(PlanState* start, PlanState* goal);

  std::vector<double> env_size_;
  double max_radius_;
  double pace_;
  PlanGrid* grid_;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >
      path_;
  std::vector<TwoDCell*> cell_path_;
};
*/

}  // namespace bewego
