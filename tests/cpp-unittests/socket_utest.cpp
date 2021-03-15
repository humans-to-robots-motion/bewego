// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/util/eigen.h>
#include <bewego/util/misc.h>
#include <bewego/util/range.h>
#include <bewego/util/socket.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <iostream>

using namespace bewego::util;
using namespace std;

bool full_precision_csv = false;

TEST(ToFromString, vector) {
  Eigen::MatrixXd m1 = Random(10);
  std::string msg = bewego::util::ToString(m1, false, true);
  Eigen::MatrixXd m2 = bewego::util::FromString(msg, 10, 1);
  EXPECT_LT((m1 - m2).norm(), 1.e-6);
}

TEST(ToFromString, matrix) {
  Eigen::MatrixXd m1(10, 10);
  for (uint32_t i = 0; i < 10; i++) {
    m1.col(i) = Random(10);
  }
  std::string msg = bewego::util::ToString(m1, false, true);
  cout << "msg : " << msg << endl;
  Eigen::MatrixXd m2 = bewego::util::FromString(msg, 10, 10);
  cout << "m2 : " << endl << endl;
  EXPECT_LT((m1 - m2).norm(), 1.e-6);
}

void TestSocket() {
  TcpClient c;
  string host = "127.0.0.1";
  int port = 5555;

  // connect to host
  c.Connect(host, port);

  // std::vector<int> v_i = range(1, 10);
  Eigen::MatrixXd v(10, 1);
  // for (uint32_t i = 0; i < v.size(); i++) {
  //   v(i, 0) = v_i[i];
  // }
  v.col(0) = Random(10);
  std::string msg = ToString(v, false, false);

  // send some data
  c.SendData(msg);

  std::string echo = c.Receive(1024);
  Eigen::MatrixXd m = FromString(msg, 10, 1);

  // receive and echo reply
  cout << "----------------------------\n\n";
  cout << m;
  cout << "\n\n----------------------------\n\n";
}

int main(int argc, char* argv[]) {
  // TestSocket();

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

  // return 0;
}