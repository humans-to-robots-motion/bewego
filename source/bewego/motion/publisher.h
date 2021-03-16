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

#pragma once

#include <bewego/util/eigen.h>
#include <bewego/util/socket.h>

#include <Eigen/Core>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace bewego {

/**
 * Class to publish a trajectory on a TCP socket
 *
 * The trajectory is simply transmited as a serialized vector
 * It is assume the server has access to q_init. Hence only the active
 * part of the trajectory is published. This is very convienent to debug
 * optimization or generally planning algorithms where the planning is
 * done in a different process as the visualization.
 */
class TrajectoryPublisher {
 public:
  TrajectoryPublisher()
      : running_(false),
        finished_(false),
        x_(Eigen::VectorXd()),
        slow_down_(false),
        t_pause_(100000) {}
  virtual ~TrajectoryPublisher() {}

  void Initialize(const std::string& host, uint32_t port,
                  const Eigen::VectorXd& q_init);

  virtual void PublishTrajectory();

  void Stop() {
    if (running_) {
      finished_ = true;
      thread_.join();
    }
  }
  void Run();

  //! Set a pause when displaying the trajectory
  void set_slow_down(bool v) { slow_down_ = v; }
  void set_t_pause(uint32_t v) { t_pause_ = v; }

  //! Set the current trajectory solution.
  void set_current_solution(const Eigen::VectorXd& x) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slow_down_) {
      std::this_thread::sleep_for(std::chrono::microseconds(t_pause_));
    }
    x_ = x;
  }

 protected:
  bool verbose_;

  std::atomic_bool running_;
  std::atomic_bool finished_;

  // must keep at 0 size at initialization
  Eigen::VectorXd x_;
  Eigen::VectorXd q_init_;

  // Client
  std::shared_ptr<util::TcpClient> tcp_client_;
  std::shared_ptr<util::Serializer> to_ascii_;

  // locks when writting in the marker
  mutable std::mutex mutex_;
  std::thread thread_;

  // Slow down
  bool slow_down_;
  uint32_t t_pause_;
};

}  // namespace bewego
