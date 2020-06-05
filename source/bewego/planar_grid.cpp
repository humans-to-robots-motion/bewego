// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include "planar_grid.h"
#include <iostream>

// #include "API/ConfigSpace/configuration.hpp"
// #include "API/Search/AStar/AStar.hpp"
// #include "API/project.hpp"
// #include "AStarPlanner.hpp"
// #include "Collision-pkg.h"
// #include "Graphic-pkg.h"
// #include "P3d-pkg.h"
// #include "hri_costspace/grid/HRICS_grid.hpp"
// #include "planner/TrajectoryOptim/Classic/smoothing.hpp"
// #include "planner/TrajectoryOptim/plannarTrajectorySmoothing.hpp"
// #include "planner/planEnvironment.hpp"

// std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> >
// path_to_draw;

using namespace std;
using namespace Eigen;

namespace bewego {

//---------------------------------------------------------------------------
// TwoDCell implementation
//---------------------------------------------------------------------------

TwoDCell::TwoDCell() {}
TwoDCell::TwoDCell(int i, Vector2d corner, TwoDGrid* grid)
    : _index(i), _corner(corner), _grid(grid) {
  //    cout << " ThreeDCell " << i << ", Cornner = "<<  _corner.at(0) <<
  //    _corner.at(1) <<  _corner.at(2) << ", Grid = " << _grid << endl;
}

TwoDCell::~TwoDCell() {}

Vector2d TwoDCell::getCenter() {
  //    cout << "getCenter()" << endl;

  Vector2d dimentions = _grid->getCellSize();

  for (int i = 0; i < dimentions.size(); i++) {
    dimentions[i] = dimentions[i] / 2 + _corner[i];
  }

  return dimentions;
}

Vector2d TwoDCell::getRandomPoint() {
  Vector2d X = Vector2d::Random();

  Matrix2d A = Matrix2d::Zero();

  A(0, 0) = _grid->getCellSize()[0] / 2;
  A(1, 1) = _grid->getCellSize()[1] / 2;
  //    A(2,2) = _grid->getCellSize()[2]/2;

  X = A * X;

  Vector2d B = this->getCenter();

  return X + B;
}

Vector2d TwoDCell::getCellSize() { return _grid->getCellSize(); }

//---------------------------------------------------------------------------
// TwoD grid implementation
//---------------------------------------------------------------------------

TwoDGrid::TwoDGrid() {}
TwoDGrid::~TwoDGrid() {
  //    for(unsigned int i=0;i<_cells.size();i++)
  //    {
  //        delete _cells.at(i);
  //    }
}

TwoDGrid::TwoDGrid(Vector2i numCell, vector<double> envSize)

{
  envSize.resize(4);
  setEnvSizeAndNumCell(numCell[0], numCell[1], envSize);
}

TwoDGrid::TwoDGrid(double samplingRate, vector<double> envSize) {
  envSize.resize(4);

  for (unsigned int i = 0; i < envSize.size(); i++) {
    cout << envSize.at(i) << " ";
  }
  cout << endl;

  if (((int)samplingRate) != 0) {
    if ((((int)(envSize.at(1) - envSize.at(0))) % (int)samplingRate) != 0) {
      cout << "TwoDGrid Warning : not good X disctretization " << endl;
    }

    if ((((int)(envSize.at(3) - envSize.at(2))) % (int)samplingRate) != 0) {
      cout << "TwoDGrid Warning : not good Y disctretization " << endl;
    }
  }

  //    _cellSize.push_back( (envSize.at(1) - envSize.at(0))/pace );
  //    _cellSize.push_back( (envSize.at(3) - envSize.at(2))/pace );

  _cellSize[0] = samplingRate;
  _cellSize[1] = samplingRate;

  _nbCellsX = (envSize.at(1) - envSize.at(0)) / samplingRate;
  _nbCellsY = (envSize.at(3) - envSize.at(2)) / samplingRate;

  cout << " _nbCellsX = " << _nbCellsX << endl;
  cout << " _nbCellsY = " << _nbCellsY << endl;

  _originCorner[0] = envSize.at(0);
  _originCorner[1] = envSize.at(2);

  cout << "_originCorner[0] = " << _originCorner[0] << endl;
  cout << "_originCorner[1] = " << _originCorner[1] << endl;
}

void TwoDGrid::setEnvSizeAndNumCell(int x, int y, std::vector<double> envSize) {
  _nbCellsX = x;
  _nbCellsY = y;

  _cellSize[0] = (envSize.at(1) - envSize.at(0)) / _nbCellsX;
  _cellSize[1] = (envSize.at(3) - envSize.at(2)) / _nbCellsY;

  _originCorner[0] = envSize.at(0);
  _originCorner[0] = envSize.at(2);
}

void TwoDGrid::createAllCells() {
  unsigned int nbCells = _nbCellsX * _nbCellsY;

  _cells.resize(nbCells);

  unsigned int x = 0;
  unsigned int y = 0;

  for (unsigned int i = 0; i < nbCells; i++) {

    TwoDCell* ptrCell = createNewCell(i, x, y);
    _cells[i] = ptrCell;

    x++;
    if (x >= _nbCellsX) {
      y++;
      x = 0;
      if (y >= _nbCellsY) {
        return;
      }
    }
  }
}

TwoDCell* TwoDGrid::getCell(const Vector2i& coord) {
  return dynamic_cast<TwoDCell*>(_cells[coord[0] + coord[1] * _nbCellsX]);
}

TwoDCell* TwoDGrid::getCell(int x, int y) {
  Vector2i coord;
  coord[0] = x;
  coord[1] = y;

  return getCell(coord);
}

TwoDCell* TwoDGrid::getCell(Vector2d point) {
  Vector2i coord;
  coord[0] = (int)floor((abs(point[0] - _originCorner[0])) / _cellSize[0]);
  coord[1] = (int)floor((abs(point[1] - _originCorner[1])) / _cellSize[1]);

  if (!isCellCoordInGrid(coord)) {
    cout << "TwoDGrid::OutBounds" << endl;
    return 0x0;
  }

  return getCell(coord);
}

TwoDCell* TwoDGrid::getCell(double* pos) {
  Vector2i coord;
  coord[0] = (int)((pos[0] - _originCorner[0]) / _cellSize[0]);
  coord[1] = (int)((pos[1] - _originCorner[1]) / _cellSize[1]);

  //    cout << "( "<<x<<" , "<<y<<" , "<<z<<" ) "<< endl;

  if (!isCellCoordInGrid(coord)) {
    cout << "ThreeDGrid::OutBounds" << endl;
    return 0x0;
  }

  return getCell(coord);
}

TwoDCell* TwoDGrid::getCell(unsigned int index) {
  return dynamic_cast<TwoDCell*>(_cells[index]);
}

bool TwoDGrid::isCellCoordInGrid(const Vector2i& coord) {
  return !(coord[0] >= ((int)_nbCellsX) || coord[1] >= ((int)_nbCellsY) ||
           coord[0] < 0 || coord[1] < 0);
}

Vector2i TwoDGrid::getCellCoord(TwoDCell* ptrCell) {
  Vector2i coord;

  int i = ptrCell->getIndex();

  coord[0] = (i / 1) % 2 - 1;  // x
  coord[1] = (i / 2) % 2 - 1;  // y

  return coord;
}

TwoDCell* TwoDGrid::createNewCell(unsigned int index, unsigned int x,
                                          unsigned int y) {
  if (index == 0) {
    return new TwoDCell(0, _originCorner, this);
  }
  TwoDCell* newCell = new TwoDCell(index, computeCellCorner(x, y), this);
  return newCell;
}

Vector2d TwoDGrid::computeCellCorner(int x, int y) {
  Vector2d corner;

  corner[0] = _originCorner[0] + x * (_cellSize[0] * 1.0);
  corner[1] = _originCorner[1] + y * (_cellSize[1] * 1.0);

  return corner;
}

int TwoDGrid::getNumberOfCells() { return _cells.size(); }

TwoDCell* TwoDGrid::getNeighbour(const Vector2i& pos, int i) {
  if (i < 0 || i > 8) {
    return 0x0;
  } else {
    if (i >= 4) i++;

    int dx = (i / 1) % 3 - 1;
    int dy = (i / 3) % 3 - 1;

    //    cout << "( "<<dx<<" , "<<dy<<" ) "<< endl;

    Vector2i coord;

    coord[0] = pos[0] + dx;
    coord[1] = pos[1] + dy;

    if (!this->isCellCoordInGrid(coord)) {
      return 0x0;
    } else {
      return getCell(coord);
    }
  }
}

Vector2d TwoDGrid::getCoordinates(TwoDCell* cell) {
  Vector2d coordinates;
  int index = cell->getIndex();
  int sizeXY = _nbCellsX * _nbCellsY;
  coordinates[2] = floor(index / sizeXY);
  coordinates[1] = floor((index - coordinates[2] * sizeXY) / _nbCellsX);
  //  coordinates[0] = floor(index - coordinates[2]*sizeXY - coordinates[1] *
  //  _nbCellsX);
  return coordinates;
}

//---------------------------------------------------------------------------
// Plan grid implementation
//---------------------------------------------------------------------------

PlanGrid::PlanGrid(double pace, vector<double> env_size,
                   bool print_cost)
    : TwoDGrid(pace, env_size),
      print_cost_(print_cost),
      use_given_bounds_(false) {
  createAllCells();
  cout << "Number total of cells = " << _nbCellsX * _nbCellsY << endl;
}

TwoDCell* PlanGrid::createNewCell(unsigned int index, unsigned int x,
                                  unsigned int y) {
  Vector2i coord;
  coord[0] = x;
  coord[1] = y;

  TwoDCell* newCell;

  if (index == 0) {
    newCell = new PlanCell(0, coord, _originCorner, this);
  } else {
    newCell = new PlanCell(index, coord, computeCellCorner(x, y), this);
  }

  return newCell;
}

void PlanGrid::reset() {
  for (int i = 0; i < int(_cells.size()); i++) {
    PlanCell* cell = dynamic_cast<PlanCell*>(_cells[i]);

    if (cell != NULL) {
      cell->resetExplorationStatus();
      cell->resetCost();
      cell->resetIsValid();
    }
  }
}

std::pair<double, double> PlanGrid::getMinMaxCost() {
  double max = std::numeric_limits<double>::min();  // TODO this is 0 ...
  double min = std::numeric_limits<double>::max();

  for (int i = 0; i < int(_cells.size()); i++) {
    double cost = dynamic_cast<PlanCell*>(_cells[i])->getCost();

    if (cost > max) max = cost;
    if (cost < min) min = cost;
  }

  std::pair<double, double> result;
  result.first = min;
  result.second = max;
  return result;
}

//---------------------------------------------------------------------------
// PlanCell Implementation
//---------------------------------------------------------------------------

PlanCell::PlanCell(int i, Vector2i coord, Vector2d corner, PlanGrid* grid)
    : TwoDCell(i, corner, grid),
      coord_(coord),
      open_(false),
      closed_(false),
      cost_is_computed_(false),
      cost_(0.0),
      is_cell_tested_(false),
      is_valid_(false) {}

double PlanCell::getCost() {
    if (cost_is_computed_)
      return cost_;
    else
      return 0.0;
}

bool PlanCell::isValid() {
  if (is_cell_tested_) {
    return is_valid_;
  }

  is_valid_ = false;

  // if (dynamic_cast<PlanGrid*>(_grid)->getRobot()->setAndUpdate(*q)) {
  //   is_valid_ = !q->isInCollision();

    //    if( !mIsValid ) {
    //      p3d_print_col_pair();
    //    }
    //    else {
    //      cout << "Valid cell for robot : " << q->getRobot()->getName() <<
    //      endl;
    //    }
  // }

  is_cell_tested_ = true;
  return is_valid_;
}

//---------------------------------------------------------------------------
// PlanState Implementation
//---------------------------------------------------------------------------

PlanState::PlanState(Vector2i cell, PlanGrid* grid) : grid_(grid) {
  cell_ = dynamic_cast<PlanCell*>(grid->getCell(cell));
}

PlanState::PlanState(PlanCell* cell, PlanGrid* grid)
    : grid_(grid), cell_(cell) {}

vector<SearchState*> PlanState::Successors(SearchState* s) {
  vector<SearchState*> newStates;
  // newStates.reserve(26);

  vector<int> remove(3);
  remove[0] = -1;
  remove[1] = -1;
  remove[2] = -1;

  Vector2i coord2 = cell_->getCoord();

  if (s) {
    Vector2i coord1 = dynamic_cast<PlanState*>(s)->cell_->getCoord();

    Vector2i coord = coord1 - coord2;

    int dir = (coord[0] + 1) + (coord[1] + 1) * 3;

    // Remove directions that go back
    switch (dir) {
      case 0:
        remove[0] = 0;
        remove[1] = 1;
        remove[2] = 3;
        break;
      case 1:
        remove[0] = 1;
        remove[1] = 0;
        remove[2] = 2;
        break;
      case 2:
        remove[0] = 2;
        remove[1] = 1;
        remove[2] = 5;
        break;
      case 3:
        remove[0] = 3;
        remove[1] = 6;
        remove[2] = 0;
        break;
      case 4:
        remove[0] = 4;
        remove[1] = 4;
        remove[2] = 4;
        break;
      case 5:
        remove[0] = 5;
        remove[1] = 8;
        remove[2] = 2;
        break;
      case 6:
        remove[0] = 6;
        remove[1] = 3;
        remove[2] = 7;
        break;
      case 7:
        remove[0] = 7;
        remove[1] = 6;
        remove[2] = 8;
        break;
      case 8:
        remove[0] = 8;
        remove[1] = 7;
        remove[2] = 5;
        break;
    };
  }

  for (int i = 0; i < 8; i++) {
    // if( i == remove[0] || i == remove[1] || i == remove[2] ){
    //       continue;
    // }

    PlanCell* neigh = dynamic_cast<PlanCell*>(grid_->getNeighbour(coord2, i));
    if (neigh != NULL) {
      newStates.push_back(new PlanState(neigh, grid_));
    }
  }

  return newStates;
}

bool PlanState::leaf() { return false; }
bool PlanState::valid() { return cell_->isValid(); }
bool PlanState::equal(SearchState* other) {
  // bool equal(false);
  PlanState* state = dynamic_cast<PlanState*>(other);

  if (cell_ != state->cell_) {
    // cout << "PlanState::equal false" << endl;
    return false;
  }

  // cout << "State::equal true" << endl;
  return true;
}

void PlanState::setClosed(std::vector<PlanState*>& closedStates,
                          std::vector<PlanState*>& openStates) {
  // cout << "State :: set Closed" <<endl;
  cell_->setClosed();
}

bool PlanState::isColsed(std::vector<PlanState*>& closedStates) {
  //    cout << "State :: get Closed" <<endl;
  return cell_->getClosed();
}

void PlanState::setOpen(std::vector<PlanState*>& openStates) {
  //     cout << "State :: set open" <<endl;
  cell_->setOpen();
}

bool PlanState::isOpen(std::vector<PlanState*>& openStates) {
  //    cout << "State :: get open" <<endl;
  return cell_->getOpen();
}

void PlanState::reset() { cell_->resetExplorationStatus(); }

void PlanState::print() {}

double PlanState::computeLength(SearchState* parent) {
  PlanState* preced = dynamic_cast<PlanState*>(parent);

  double g;

  Vector2d pos1 = cell_->getCenter();
  Vector2d pos2 = preced->cell_->getCenter();

/**
  if (ENV.getBool(Env::isCostSpace)) {
    confPtr_t q1 = grid_->getRobot()->getNewConfig();
    confPtr_t q2 = grid_->getRobot()->getNewConfig();
    (*q1)[6] = pos1[0];
    (*q2)[6] = pos2[0];
    (*q1)[7] = pos1[1];
    (*q2)[7] = pos2[1];
    LocalPath path(q2, q1);
    g = preced->g() + path.cost();
  } else {
    g = preced->g() + (pos1 - pos2).norm();
    cell_->setCost(g);
  }
    */
  return g;

}

double PlanState::computeHeuristic(SearchState* parent, SearchState* goal) {
  //    return 0.0;
  // if (!ENV.getBool(Env::isCostSpace)) {
  //   PlanState* state = dynamic_cast<PlanState*>(goal);
  //   Vector2d posGoal = state->cell_->getCenter();
  //   Vector2d posThis = cell_->getCenter();
  //   return (posGoal - posThis).norm();
  // } 
  return 0.0;
}

/**
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

AStarPlanner::AStarPlanner()  {
  // cout << "Create planner for robot : " << R->getName() << endl;
  pace_ = 1.0;

  // double diagonal = std::sqrt( std::pow(m_envSize[1]-m_envSize[0], 2 ) +
  // std::pow(m_envSize[3]-m_envSize[2] , 2 ) ); double pace = 0.20;
  //    init();
}

AStarPlanner::~AStarPlanner() {
  //    if( grid_ == API_activeGrid )
  //        API_activeGrid = NULL;

  //    delete grid_;
}

unsigned int AStarPlanner::init() {
  env_size_ = global_Project->getActiveScene()->getBounds();
  env_size_.resize(4);

  cout << "pace : " << pace_ << " meters" << endl;

  grid_ = new PlanGrid(_Robot, pace_, env_size_);

  if (API_activeGrid != NULL) {
    delete API_activeGrid;
  }
  API_activeGrid = grid_;

  return 1;
}

void AStarPlanner::reset() { grid_->reset(); }

bool AStarPlanner::solveAStar(PlanState* start, PlanState* goal) {
  bool path_exists = true;
  path_.clear();

  // Change the way AStar is computed to go down
  if (start->getCell()->getCost() < goal->getCell()->getCost()) {
    AStar* search = new AStar(start);
    vector<State*> path = search->solve(goal);

    if (path.size() == 0) {
      path_.clear();
      cell_path_.clear();
      path_exists = false;
      return path_exists;
    }

    for (unsigned int i = 0; i < path.size(); i++) {
      TwoDCell* cell = dynamic_cast<PlanState*>(path[i])->getCell();
      path_.push_back(cell->getCenter());
      cell_path_.push_back(cell);
    }
  } else {
    AStar* search = new AStar(goal);
    vector<State*> path = search->solve(start);

    if (path.size() == 0) {
      path_.clear();
      cell_path_.clear();
      path_exists = false;
      return path_exists;
    }

    for (int i = path.size() - 1; i >= 0; i--) {
      PlanCell* cell = dynamic_cast<PlanState*>(path[i])->getCell();
      path_.push_back(cell->getCenter());
      cell_path_.push_back(cell);
    }
  }

  return path_exists;
}

bool AStarPlanner::computeAStarIn2DGrid(Vector2d source, Vector2d target) {
  PlanCell* startCell = dynamic_cast<PlanCell*>(grid_->getCell(source));
  if (startCell == NULL) {
    cout << "start (" << source.transpose() << ") not in grid" << endl;
    return false;
  }

  PlanCell* goalCell = dynamic_cast<PlanCell*>(grid_->getCell(target));
  if (goalCell == NULL) {
    cout << "goal (" << target.transpose() << ") not in grid" << endl;
    return false;
  }

  Vector2i startCoord = startCell->getCoord();
  cout << "Start Pos = (" << source[0] << " , " << source[1] << ")" << endl;
  cout << "Start Coord = (" << startCoord[0] << " , " << startCoord[1] << ")"
       << endl;

  Vector2i goalCoord = goalCell->getCoord();
  cout << "Goal Pos = (" << target[0] << " , " << target[1] << ")" << endl;
  cout << "Goal Coord = (" << goalCoord[0] << " , " << goalCoord[1] << ")"
       << endl;

  if (startCoord == goalCoord) {
    cout << " no planning as cells are identical" << endl;
    return false;
  }

  PlanState* start = new PlanState(startCell, grid_);
  PlanState* goal = new PlanState(goalCell, grid_);
  if (start == NULL || goal == NULL) {
    cout << "Start or goal == NULL" << endl;
    return false;
  }

  if (solveAStar(start, goal)) {
    double SumOfCost = 0.0;
    for (int i = 0; i < int(path_.size()); i++) {
      // cout << "Cell "<< i <<" = " << endl << path_[i] << endl;
      SumOfCost += dynamic_cast<PlanCell*>(cell_path_[i])->getCost();
    }
    cout << " SumOfCost = " << SumOfCost << endl;
    return true;
  } else {
    return false;
  }
}

Trajectory* AStarPlanner::computeRobotTrajectory(confPtr_t source,
                                                 confPtr_t target) {
  confPtr_t q = _Robot->getCurrentPos();
  (*q)[6] = 0;
  (*q)[7] = 0;

  Vector2d x1, x2;

  x1[0] = (*source)[6];
  x1[1] = (*source)[7];

  x2[0] = (*target)[6];
  x2[1] = (*target)[7];

  _Robot->setAndUpdate(*source);

  if (computeAStarIn2DGrid(x1, x2)) {
    Trajectory* traj = new Trajectory(_Robot);

    traj->push_back(source);

    for (int i = 0; i < int(path_.size()); i++) {
      confPtr_t q = _Robot->getCurrentPos();
      (*q)[6] = path_[i][0];
      (*q)[7] = path_[i][1];

      traj->push_back(q);
    }
    traj->push_back(target);
    traj->computeSubPortionIntergralCost(traj->getCourbe());
    traj->replaceP3dTraj();
    _Robot->setAndUpdate(*q);
    return traj;
  } else {
    _Robot->setAndUpdate(*q);
    return NULL;
  }
}

void AStarPlanner::draw() {
  for (int i = 0; i < int(path_.size()) - 1; i++) {
    glLineWidth(3.);
    g3d_drawOneLine(path_[i][0], path_[i][1], 0.4, path_[i + 1][0],
                    path_[i + 1][1], 0.4, Yellow, NULL);
    glLineWidth(1.);
  }
}

void AStarPlanner::allow_smoothing(bool state) {
  PlanEnv->setBool(PlanParam::env_createTrajs, state);
}
*/

}
