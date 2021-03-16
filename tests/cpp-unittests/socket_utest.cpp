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
  bool verbose = false;
  Eigen::MatrixXd m1(10, 10);
  for (uint32_t i = 0; i < 10; i++) {
    m1.col(i) = Random(10);
  }
  std::string msg = bewego::util::ToString(m1, false, true);
  Eigen::MatrixXd m2 = bewego::util::FromString(msg, 10, 10);
  if (verbose) {
    cout << "m1 : " << endl << m1 << endl;
    cout << "m2 : " << endl << m2 << endl;
  }
  EXPECT_LT((m1 - m2).norm(), 1.e-6);
}

TEST(ToFromString, matrix_wide) {
  bool verbose = false;
  Eigen::MatrixXd m1(2, 10);
  for (uint32_t i = 0; i < 10; i++) {
    m1.col(i) = Random(2);
  }

  std::string msg = bewego::util::ToString(m1, false, true);
  Eigen::MatrixXd m2 = bewego::util::FromString(msg, 2, 10);
  if (verbose) {
    cout << "m1 : " << endl << m1 << endl;
    cout << "m2 : " << endl << m2 << endl;
  }
  EXPECT_LT((m1 - m2).norm(), 1.e-6);
}

TEST(Serializer, matrices) {
  bool verbose = false;
  double epsilon_mat = 1e-12;

  Eigen::MatrixXd ma1, ma2;
  std::string msg;
  bewego::util::Serializer s;

  ma1 = Eigen::MatrixXd::Random(2, 10);
  msg = s.Serialize(ma1);
  ma2 = s.Deserialize(msg);
  if (verbose) {
    cout << "ma1 : " << endl << ma1 << endl;
    cout << "ma2 : " << endl << ma2 << endl;
  }
  EXPECT_EQ(ma1.rows(), ma2.rows());
  EXPECT_LT((ma1 - ma2).norm(), epsilon_mat);

  ma1 = Eigen::MatrixXd::Random(12, 3);
  msg = s.Serialize(ma1);
  ma2 = s.Deserialize(msg);
  if (verbose) {
    cout << "ma1 : " << endl << ma1 << endl;
    cout << "ma2 : " << endl << ma2 << endl;
  }
  EXPECT_EQ(ma1.rows(), ma2.rows());
  EXPECT_LT((ma1 - ma2).norm(), epsilon_mat);

  ma1 = Eigen::MatrixXd::Random(13, 1);
  msg = s.Serialize(ma1);
  ma2 = s.Deserialize(msg);
  if (verbose) {
    cout << "ma1 : " << endl << ma1 << endl;
    cout << "ma2 : " << endl << ma2 << endl;
  }
  EXPECT_EQ(ma1.rows(), ma2.rows());
  EXPECT_LT((ma1 - ma2).norm(), epsilon_mat);

  ma1 = Eigen::MatrixXd::Random(1, 11);
  msg = s.Serialize(ma1);
  ma2 = s.Deserialize(msg);
  if (verbose) {
    cout << "ma1 : " << endl << ma1 << endl;
    cout << "ma2 : " << endl << ma2 << endl;
  }
  EXPECT_EQ(ma1.rows(), ma2.rows());
  EXPECT_LT((ma1 - ma2).norm(), epsilon_mat);
}

void TestSocket() {
  TcpClient c;
  string host = "127.0.0.1";
  int port = 5555;
  std::string echo, msg;
  Eigen::MatrixXd matrix;
  bewego::util::Serializer s;

  // connect to host
  c.Connect(host, port);

  for (uint32_t i = 0; i < 100; i++) {
    // std::vector<int> v_i = range(1, 10);
    matrix = Eigen::MatrixXd::Random(3, 10);
    cout << i << " * matrix : " << endl << matrix << endl;

    msg = s.Serialize(matrix);

    // send some data
    c.SendData(msg);
    echo = c.Receive(4);
    if (echo != "ackn") {
      cerr << "Error in trajectory transmission echo (received: " << echo << ")"
           << endl;
      break;
    }
  }

  c.SendData("end");
  echo = c.Receive(4);
  if (echo != "done") {
    cerr << "Error in close transmission echo : " << echo << endl;
  }

  // std::string echo = c.Receive(1024);
  // Eigen::MatrixXd m = s.Deserialize(echo);

  // receive and echo reply
  // cout << "----------------------------\n\n";
  // cout << m;
  // cout << "\n\n----------------------------\n\n";

  cout << "Close socket..." << endl;
  c.Close();
  cout << "Done." << endl;
}

int main(int argc, char* argv[]) {
  // TestSocket();
  // return 0;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}