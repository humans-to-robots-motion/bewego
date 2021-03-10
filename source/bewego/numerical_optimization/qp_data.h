/*
 * Copyright (c) 2019
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
 *                                              Jim Mainprice Tue 18 Nov 2019
 */
#pragma once

#include <Eigen/Core>

namespace bewego {

struct QPData {
  QPData() { Initialize(); }
  void Initialize() {
    A = Eigen::MatrixXd(5, 10);
    A << -0.0171265, -0.907998, -0.619952, 0.383558, 0.172114, -0.679482,
        0.201939, -0.429128, 0.734167, 0.11237, 0.155115, -0.72696, 0.466806,
        0.455254, 0.715855, -0.0523114, -0.00741706, -0.352406, -0.858782,
        0.595856, -0.974063, -0.024194, -0.394081, -0.538017, -0.632318,
        0.801667, -0.65848, -0.890467, 0.458223, 0.553256, 0.928971, -0.628587,
        0.687239, -0.459026, 0.631935, -0.388302, 0.926616, -0.0704659,
        -0.649046, 0.570073, -0.780443, -0.653693, 0.429011, -0.857014, 0.93457,
        -0.193717, -0.37237, -0.320061, -0.511566, -0.779255;

    a = Eigen::VectorXd(5);
    a << -0.935965, -0.760776, -0.358538, 0.0565676, 0.731787;

    B = Eigen::MatrixXd(2, 10);
    B << -0.861953, -0.311752, -0.116767, -0.099029, -0.235892, 0.798484,
        0.292092, 0.972071, 0.364634, -0.801771, -0.847523, 0.39233, -0.499917,
        -0.380927, -0.64516, 0.117587, -0.80306, -0.396123, 0.406807, 0.632503;

    b = Eigen::VectorXd(2);
    b << 0.480174, 0.277658;

    c = 2.;

    d = Eigen::VectorXd(10);
    d << -0.464597, -0.480692, -0.983101, -0.977804, 0.0514354, 0.474272,
        -0.914043, -0.323142, 0.958868, -0.307016;

    H = Eigen::MatrixXd(10, 10);
    H << 3.43294, 1.04819, 2.20076, -0.426663, -0.268562, 0.0717742, 1.03703,
        0.288651, 0.0236153, 0.385183, 1.04819, 3.78208, -0.00189074, 0.496562,
        0.125932, 0.822186, 0.934085, -0.488965, -0.354886, 2.67397, 2.20076,
        -0.00189074, 4.25835, -1.27293, 0.441948, -0.354877, 0.0907788,
        -0.721566, 0.486566, 1.06365, -0.426663, 0.496562, -1.27293, 3.27162,
        -1.06405, -0.905424, -0.892559, 0.747805, -2.13337, 1.89636, -0.268562,
        0.125932, 0.441948, -1.06405, 1.48172, -0.419329, 0.858664, -0.206424,
        0.383331, -0.168021, 0.0717742, 0.822186, -0.354877, -0.905424,
        -0.419329, 3.21938, -1.58184, -0.587143, 2.57313, -0.609366, 1.03703,
        0.934085, 0.0907788, -0.892559, 0.858664, -1.58184, 3.82726, 0.869463,
        -1.10754, 0.0738853, 0.288651, -0.488965, -0.721566, 0.747805,
        -0.206424, -0.587143, 0.869463, 3.80227, -0.521151, -0.275371,
        0.0236153, -0.354886, 0.486566, -2.13337, 0.383331, 2.57313, -1.10754,
        -0.521151, 4.32092, -1.33462, 0.385183, 2.67397, 1.06365, 1.89636,
        -0.168021, -0.609366, 0.0738853, -0.275371, -1.33462, 4.05504;
  }
  Eigen::MatrixXd A;
  Eigen::VectorXd a;
  Eigen::MatrixXd B;
  Eigen::VectorXd b;
  double c;
  Eigen::VectorXd d;
  Eigen::MatrixXd H;
};

}  // namespace bewego
