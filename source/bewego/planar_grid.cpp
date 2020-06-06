// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include "planar_grid.h"
#include "chrono.h"

#include <iostream>

using namespace std;
using namespace Eigen;

namespace bewego {

//------------------------------------------------------------------------------
// TwoDCell implementation
//------------------------------------------------------------------------------

TwoDCell::TwoDCell() {}
TwoDCell::TwoDCell(int i, const Vector2d& corner, TwoDGrid* grid)
    : index_(i), corner_(corner), grid_(grid) {}
TwoDCell::~TwoDCell() {}

Vector2d TwoDCell::Center() const {
  Vector2d dimentions = grid_->cell_size();
  for (int i = 0; i < dimentions.size(); i++) {
    dimentions[i] = .5 * dimentions[i] + corner_[i];
  }
  return dimentions;
}

Vector2d TwoDCell::RandomPoint() const {
  Vector2d x = Vector2d::Random();
  Matrix2d A = Matrix2d::Zero();
  Vector2d b = Center();
  A(0, 0) = grid_->cell_size().x();
  A(1, 1) = grid_->cell_size().y();
  return .5 * A * x + b;
}

Vector2d TwoDCell::cell_size() const { return grid_->cell_size(); }

//------------------------------------------------------------------------------
// TwoD grid implementation
//------------------------------------------------------------------------------

TwoDGrid::TwoDGrid() {}
TwoDGrid::~TwoDGrid() {}
TwoDGrid::TwoDGrid(const Vector2i& numCell, const vector<double>& env_size) {
  assert(env_size.size() == 4);
  setEnvSizeAndNumCell(numCell[0], numCell[1], env_size);
}

TwoDGrid::TwoDGrid(double samplingRate, const vector<double>& env_size) {
  assert(env_size.size() == 4);
  if (((int)samplingRate) != 0) {
    if ((((int)(env_size.at(1) - env_size.at(0))) % (int)samplingRate) != 0) {
      cerr << "TwoDGrid Warning : not good X disctretization " << endl;
    }

    if ((((int)(env_size.at(3) - env_size.at(2))) % (int)samplingRate) != 0) {
      cerr << "TwoDGrid Warning : not good Y disctretization " << endl;
    }
  }

  cell_size_[0] = samplingRate;
  cell_size_[1] = samplingRate;

  nb_cells_x_ = (env_size.at(1) - env_size.at(0)) / samplingRate;
  nb_cells_y_ = (env_size.at(3) - env_size.at(2)) / samplingRate;

  origin_corner_[0] = env_size.at(0);
  origin_corner_[1] = env_size.at(2);
}

void TwoDGrid::setEnvSizeAndNumCell(int x, int y,
                                    std::vector<double> env_size_) {
  nb_cells_x_ = x;
  nb_cells_y_ = y;

  cell_size_[0] = (env_size_.at(1) - env_size_.at(0)) / nb_cells_x_;
  cell_size_[1] = (env_size_.at(3) - env_size_.at(2)) / nb_cells_y_;

  origin_corner_[0] = env_size_.at(0);
  origin_corner_[0] = env_size_.at(2);
}

void TwoDGrid::createAllCells() {
  uint32_t nb_cells = nb_cells_x_ * nb_cells_y_;
  cells_.resize(nb_cells);
  uint32_t x = 0;
  uint32_t y = 0;
  for (uint32_t i = 0; i < nb_cells; i++) {
    cells_[i] = createNewCell(i, x, y);
    x++;
    if (x >= nb_cells_x_) {
      y++;
      x = 0;
      if (y >= nb_cells_y_) {
        return;
      }
    }
  }
}

TwoDCell* TwoDGrid::getCell(uint32_t index) const {
  return dynamic_cast<TwoDCell*>(cells_[index]);
}

TwoDCell* TwoDGrid::getCell(const Vector2i& coord) const {
  return dynamic_cast<TwoDCell*>(cells_[coord[0] + coord[1] * nb_cells_x_]);
}

TwoDCell* TwoDGrid::getCell(uint32_t x, uint32_t y) const {
  Vector2i coord;
  coord[0] = x;
  coord[1] = y;
  return getCell(coord);
}

TwoDCell* TwoDGrid::getCell(const Vector2d& point) const {
  Vector2i coord;
  coord[0] = (int)floor((abs(point[0] - origin_corner_[0])) / cell_size_[0]);
  coord[1] = (int)floor((abs(point[1] - origin_corner_[1])) / cell_size_[1]);
  if (!isCellCoordInGrid(coord)) {
    cout << "TwoDGrid::OutBounds" << endl;
    return 0x0;
  }

  return getCell(coord);
}

TwoDCell* TwoDGrid::getCell(double* pos) const {
  Vector2i coord;
  coord[0] = (int)((pos[0] - origin_corner_[0]) / cell_size_[0]);
  coord[1] = (int)((pos[1] - origin_corner_[1]) / cell_size_[1]);

  if (!isCellCoordInGrid(coord)) {
    cout << "ThreeDGrid::OutBounds" << endl;
    return 0x0;
  }

  return getCell(coord);
}

bool TwoDGrid::isCellCoordInGrid(const Vector2i& coord) const {
  return !(coord[0] >= ((int)nb_cells_x_) || coord[1] >= ((int)nb_cells_y_) ||
           coord[0] < 0 || coord[1] < 0);
}

Vector2i TwoDGrid::getCellCoord(TwoDCell* cell) const {
  Vector2i coord;

  uint32_t i = cell->index();

  coord[0] = (i / 1) % 2 - 1;  // x
  coord[1] = (i / 2) % 2 - 1;  // y

  return coord;
}

TwoDCell* TwoDGrid::createNewCell(uint32_t index, uint32_t x, uint32_t y) {
  if (index == 0) {
    return new TwoDCell(0, origin_corner_, this);
  }
  TwoDCell* newCell = new TwoDCell(index, computeCellCorner(x, y), this);
  return newCell;
}

Vector2d TwoDGrid::computeCellCorner(uint32_t x, uint32_t y) {
  Vector2d corner;

  corner[0] = origin_corner_[0] + x * (cell_size_[0] * 1.0);
  corner[1] = origin_corner_[1] + y * (cell_size_[1] * 1.0);

  return corner;
}

uint32_t TwoDGrid::getNumberOfCells() const { return cells_.size(); }

TwoDCell* TwoDGrid::getNeighbour(const Vector2i& pos, uint32_t i) const {
  if (i < 0 || i > 8) {
    return 0x0;
  } else {
    if (i >= 4) i++;

    int dx = (i / 1) % 3 - 1;
    int dy = (i / 3) % 3 - 1;

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

Vector2d TwoDGrid::getCoordinates(TwoDCell* cell) const {
  Vector2d coordinates;
  int index = cell->index();
  int sizeXY = nb_cells_x_ * nb_cells_y_;
  coordinates[2] = floor(index / sizeXY);
  coordinates[1] = floor((index - coordinates[2] * sizeXY) / nb_cells_x_);
  return coordinates;
}

//------------------------------------------------------------------------------
// Plan grid implementation
//------------------------------------------------------------------------------

PlanGrid::PlanGrid(double pace, vector<double> env_size, bool print_cost)
    : TwoDGrid(pace, env_size),
      print_cost_(print_cost),
      use_given_bounds_(false) {
  createAllCells();
  cout << "Number total of cells = " << nb_cells_x_ * nb_cells_y_ << endl;
}

TwoDCell* PlanGrid::createNewCell(uint32_t index, uint32_t x, uint32_t y) {
  Vector2i coord;
  coord[0] = x;
  coord[1] = y;

  TwoDCell* newCell;

  if (index == 0) {
    newCell = new PlanCell(0, coord, origin_corner_, this);
  } else {
    newCell = new PlanCell(index, coord, computeCellCorner(x, y), this);
  }

  return newCell;
}

void PlanGrid::Reset() {
  for (size_t i = 0; i < cells_.size(); i++) {
    PlanCell* cell = dynamic_cast<PlanCell*>(cells_[i]);
    if (cell != NULL) {
      cell->resetExplorationStatus();
      cell->resetCost();
      cell->resetIsValid();
    }
  }
}

void PlanGrid::setCostBounds(double min, double max) {
    use_given_bounds_ = true;
    min_cost_ = min;
    max_cost_ = max;
}

void PlanGrid::SetCosts(const Eigen::MatrixXd& cost) {
  assert(cost.rows() == nb_cells_x_);
  assert(cost.cols() == nb_cells_y_);

  for (size_t i = 0; i < nb_cells_x_; i++) {
    for (size_t j = 0; j < nb_cells_y_; j++) {
        size_t index = i + j * nb_cells_x_;
        static_cast<PlanCell*>(cells_[index])->setCost(cost(i, j));
      }
    }
}

std::pair<double, double> PlanGrid::getMinMaxCost() {
  double max = std::numeric_limits<double>::min();  // TODO this is 0 ...
  double min = std::numeric_limits<double>::max();

  for (size_t i = 0; i < cells_.size(); i++) {
    double cost = dynamic_cast<PlanCell*>(cells_[i])->getCost();
    if (cost > max) max = cost;
    if (cost < min) min = cost;
  }

  std::pair<double, double> result;
  result.first = min;
  result.second = max;
  return result;
}

//------------------------------------------------------------------------------
// PlanCell Implementation
//------------------------------------------------------------------------------

PlanCell::PlanCell(int i, Vector2i coord, Vector2d corner, PlanGrid* grid)
    : TwoDCell(i, corner, grid),
      coord_(coord),
      open_(false),
      closed_(false),
      cost_is_computed_(false),
      cost_(0.0),
      is_cell_tested_(false),
      is_valid_(true) {}

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
  return true;
}

//------------------------------------------------------------------------------
// PlanState Implementation
//------------------------------------------------------------------------------

PlanState::PlanState(Vector2i cell, PlanGrid* grid) : grid_(grid), distance_cost_(false) {
  cell_ = dynamic_cast<PlanCell*>(grid->getCell(cell));
}

PlanState::PlanState(PlanCell* cell, PlanGrid* grid)
    : grid_(grid), cell_(cell), distance_cost_(false) {}

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
  PlanState* state = dynamic_cast<PlanState*>(other);
  if (cell_ != state->cell_) {
    return false;
  }
  return true;
}

void PlanState::setClosed(std::vector<PlanState*>& closedStates,
                          std::vector<PlanState*>& openStates) {
  cell_->setClosed();
}

bool PlanState::isColsed(std::vector<PlanState*>& closedStates) {
  return cell_->getClosed();
}

void PlanState::setOpen(std::vector<PlanState*>& openStates) {
  cell_->setOpen();
}

bool PlanState::isOpen(std::vector<PlanState*>& openStates) {
  return cell_->getOpen();
}

void PlanState::reset() { cell_->resetExplorationStatus(); }

double PlanState::Length(SearchState* parent) {
  PlanState* preced = dynamic_cast<PlanState*>(parent);
  if(distance_cost_) {
      Vector2d pos1 = cell_->Center();
      Vector2d pos2 = preced->cell_->Center();
      return preced->g() + (pos1 - pos2).norm();
  }
  return preced->g() + cell_->getCost();
}

double PlanState::Heuristic(SearchState* parent, SearchState* goal) {
  if(distance_cost_) {
    PlanState* state = dynamic_cast<PlanState*>(goal);
    Vector2d posGoal = state->cell_->Center();
    Vector2d posThis = cell_->Center();
    return (posGoal - posThis).norm();
  }
  return 0;
}

//------------------------------------------------------------------------------
// A star problem implementation
//------------------------------------------------------------------------------

AStarProblem::AStarProblem() {
  pace_ = .05;
  env_size_ = {0, 1, 0, 1};
}
AStarProblem::~AStarProblem() {}

void AStarProblem::InitGrid() {
  assert(env_size_.size() == 4);
  ChronoOn();
  grid_ = std::make_shared<PlanGrid>(pace_, env_size_);
  ChronoPrint("Init grid");
  ChronoOff();
}

void AStarProblem::InitCosts(const Eigen::MatrixXd& cost) {
  assert(grid_.get() != nullptr);
  ChronoOn();
  grid_->SetCosts(cost);
  ChronoPrint("Set costs");
  ChronoOff();
}


Eigen::MatrixXi AStarProblem::PathCoordinates() const {
  Eigen::MatrixXi path_coordinates(path_.size(), 2);
  for(size_t i=0; i<cell_path_.size(); i++) {
    path_coordinates.row(i) = cell_path_[i]->getCoord();
  }
  return path_coordinates;
}

void AStarProblem::Reset() { grid_->Reset(); }

bool AStarProblem::SearchPath(PlanState* start, PlanState* goal) {
  bool path_exists = true;
  path_.clear();

  // Change the way AStar is computed to go down
  if (start->getCell()->getCost() < goal->getCell()->getCost()) {
    AStar* problem = new AStar(start);
    vector<SearchState*> path = problem->Solve(goal);

    if (path.size() == 0) {
      path_.clear();
      cell_path_.clear();
      path_exists = false;
      return path_exists;
    }

    for (unsigned int i = 0; i < path.size(); i++) {
      PlanCell* cell = dynamic_cast<PlanState*>(path[i])->getCell();
      path_.push_back(cell->Center());
      cell_path_.push_back(cell);
    }
  } else {
    AStar* problem = new AStar(goal);
    vector<SearchState*> path = problem->Solve(start);

    if (path.size() == 0) {
      cell_path_.clear();
      path_exists = false;
      return path_exists;
    }

    for (int i = path.size() - 1; i >= 0; i--) {
      PlanCell* cell = dynamic_cast<PlanState*>(path[i])->getCell();
      path_.push_back(cell->Center());
      cell_path_.push_back(cell);
    }
  }

  return path_exists;
}

bool AStarProblem::Solve(
    const Vector2d& source, 
    const Vector2d& target) {
  
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

  Eigen::Vector2i s = startCell->getCoord();
  Eigen::Vector2i g = goalCell->getCoord();
  return Solve(s, g);
}

bool AStarProblem::Solve(
    const Vector2i& start_coord, 
    const Vector2i& goal_coord) {

  ChronoOn();
  
  PlanCell* startCell = dynamic_cast<PlanCell*>(grid_->getCell(start_coord));
  if (startCell == NULL) {
    cout << "start (" << start_coord.transpose() << ") not in grid" << endl;
    return false;
  }

  PlanCell* goalCell = dynamic_cast<PlanCell*>(grid_->getCell(goal_coord));
  if (goalCell == NULL) {
    cout << "goal (" << goal_coord.transpose() << ") not in grid" << endl;
    return false;
  }

  cout << "Start Coord = (" << start_coord.transpose() << ")" << endl;
  cout << "Goal Coord = (" << goal_coord.transpose() << ")" << endl;

  if (start_coord.x() == goal_coord.x() && start_coord.y() == goal_coord.y()) {
    cout << " no planning as cells are identical" << endl;
    return false;
  }

  PlanState* start = new PlanState(startCell, grid_.get());
  PlanState* goal = new PlanState(goalCell, grid_.get());
  if (start == NULL || goal == NULL) {
    cout << "Start or goal == NULL" << endl;
    return false;
  }

  bool success = SearchPath(start, goal);

  ChronoPrint("AStar solved");
  ChronoOff();

  if (success) {
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

}  // namespace bewego
