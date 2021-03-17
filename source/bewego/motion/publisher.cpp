/**
 * Copyright (c) 2015
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
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/motion/publisher.h>
#include <bewego/motion/trajectory.h>

#include <chrono>

using std::cerr;
using std::cout;
using std::endl;

using namespace bewego;
using namespace bewego::util;

void TrajectoryPublisher::set_current_solution(const Eigen::VectorXd& x) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (slow_down_) {
    cout << "sleep..." << endl;
    std::this_thread::sleep_for(std::chrono::microseconds(t_pause_));
  }
  cout << "set current solution (" << ++ith_ << ")," << t_pause_ << endl;
  x_ = x;
}

void TrajectoryPublisher::PublishTrajectory() {
  if (x_.size() == 0) {
    return;
  }
  std::shared_ptr<Trajectory> trajectory;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    trajectory = std::make_shared<Trajectory>(q_init_, x_);
  }
  Eigen::VectorXd x_active = trajectory->ActiveSegment();
  cout << x_active.transpose() << endl;
  std::string msg = to_ascii_->Serialize(x_active);
  tcp_client_->SendData(msg);
  std::string echo = tcp_client_->Receive(4);
  if (echo != "ackn") {
    cerr << "Error in trajectory transmission echo (received: " << echo << ")"
         << endl;
    running_ = true;
    // throw std::runtime_error("TCP communication with server error");
  }
}

void TrajectoryPublisher::Initialize(const std::string& host, uint32_t port,
                                     const Eigen::VectorXd& q_init) {
  cout << __PRETTY_FUNCTION__ << endl;
  q_init_ = q_init;
  host_ = host;
  port_ = port;
  thread_ = std::thread(std::bind(&TrajectoryPublisher::Run, this));
}

void TrajectoryPublisher::Stop() {
  if (running_) {
    finished_ = true;
    thread_.join();
  }
}

void TrajectoryPublisher::Close() {
  tcp_client_->SendData("end");
  std::string echo = tcp_client_->Receive(4);
  if (echo != "done") {
    cerr << "Error in close transmission echo : " << echo << endl;
  }
  tcp_client_->Close();
  cout << "tcp server closed." << endl;
}

void TrajectoryPublisher::Run() {
  cout << __PRETTY_FUNCTION__ << endl;

  tcp_client_ = std::make_shared<TcpClient>();
  if (!tcp_client_->Connect(host_, port_)) {
    throw std::runtime_error(
        "Trajectory publisher Could not connect to server");
  }

  using namespace std::chrono;
  auto rate = milliseconds(200);
  auto start = steady_clock::now();
  auto next = steady_clock::now();
  std::chrono::duration<double> elapsed_seconds;

  running_ = true;
  finished_ = false;

  while (!finished_) {
    elapsed_seconds = steady_clock::now() - start;
    cout << "publish : " << elapsed_seconds.count() << endl;
    PublishTrajectory();

    // delay until time to iterate again
    next += rate;
    std::this_thread::sleep_until(next);
  }
  running_ = false;
  Close();
  cout << "stop publishing trajectory." << endl;
}
