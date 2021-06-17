/*
 * Copyright (c) 2021
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
 *                                               Jim Mainprice Mon 14 June 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <bewego/motion/forward_kinematics.h>

namespace bewego {

inline std::shared_ptr<KinematicChain> CreateThreeDofPlanarManipulator(
    bool last_fixed = false) {
  std::vector<RigidBodyInfo> bodies(4);

  // - parent : base
  bodies[0].name = "link1";
  bodies[0].joint_name = "base_to_link1";
  bodies[0].joint_type = 0;
  bodies[0].dof_lower_limit = -3.15;
  bodies[0].dof_upper_limit = 3.15;
  bodies[0].local_in_prev.row(0) << 1., 0., 0., 0.;
  bodies[0].local_in_prev.row(1) << 0., 1., 0., 0.;
  bodies[0].local_in_prev.row(2) << 0., 0., 1., 0.1;
  bodies[0].local_in_prev.row(3) << 0., 0., 0., 1.;
  bodies[0].joint_axis_in_local << 0., 0., 1.;

  // - parent : link1
  bodies[1].name = "link2";
  bodies[1].joint_name = "link1_to_link2";
  bodies[1].joint_type = 0;
  bodies[1].dof_lower_limit = -3.15;
  bodies[1].dof_upper_limit = 3.15;
  bodies[1].local_in_prev.row(0) << 1., 0., 0., 1.;
  bodies[1].local_in_prev.row(1) << 0., 1., 0., 0.;
  bodies[1].local_in_prev.row(2) << 0., 0., 1., 0.;
  bodies[1].local_in_prev.row(3) << 0., 0., 0., 1.;
  bodies[1].joint_axis_in_local << 0., 0., 1.;

  // - parent : link2
  bodies[2].name = "link3";
  bodies[2].joint_name = "link2_to_link3";
  bodies[2].joint_type = 0;
  bodies[2].dof_lower_limit = -3.15;
  bodies[2].dof_upper_limit = 3.15;
  bodies[2].local_in_prev.row(0) << 1., 0., 0., 1.;
  bodies[2].local_in_prev.row(1) << 0., 1., 0., 0.;
  bodies[2].local_in_prev.row(2) << 0., 0., 1., 0.;
  bodies[2].local_in_prev.row(3) << 0., 0., 0., 1.;
  bodies[2].joint_axis_in_local << 0., 0., 1.;

  // - parent : link3
  bodies[3].name = "end";
  bodies[3].joint_name = "link3_to_end";
  bodies[3].joint_type = last_fixed ? 4 : 0;  // 4 = FIXED
  bodies[3].dof_lower_limit = -3.15;
  bodies[3].dof_upper_limit = 3.15;
  bodies[3].local_in_prev.row(0) << 1., 0., 0., 1.;
  bodies[3].local_in_prev.row(1) << 0., 1., 0., 0.;
  bodies[3].local_in_prev.row(2) << 0., 0., 1., 0.;
  bodies[3].local_in_prev.row(3) << 0., 0., 0., 1.;
  bodies[3].joint_axis_in_local << 0., 0., 1.;

  return std::make_shared<KinematicChain>(bodies);
}

}  // namespace bewego