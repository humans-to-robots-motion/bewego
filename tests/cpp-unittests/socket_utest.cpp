// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/util/socket.h>

using namespace bewego::util;
using namespace std;

int main(int argc, char *argv[]) {
  TcpClient c;
  string host = "127.0.0.1";

  // connect to host
  c.Connect(host, 5555);

  // send some data
  c.SendData("Hello");

  // receive and echo reply
  // cout << "----------------------------\n\n";
  // cout << c.Receive(1024);
  // cout << "\n\n----------------------------\n\n";

  // done
  return 0;
}