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
 *                                              Jim Mainprice Sun 14 Mar 2021
 */
#pragma once

#include <arpa/inet.h>   //inet_addr
#include <netdb.h>       //hostent
#include <stdio.h>       //printf
#include <string.h>      //strlen
#include <sys/socket.h>  //socket

#include <iostream>  //cout
#include <string>    //string

namespace bewego {
namespace util {

class TcpClient {
 public:
  TcpClient();
  bool Connect(std::string, int);
  bool SendMessage(std::string data);
  bool SendData(std::string data);
  std::string Receive(int);
  void Close();

 private:
  bool verbose_;
  int sock;
  std::string address;
  std::string response_data = "";
  int port;
  struct sockaddr_in server;
};

}  // namespace util
}  // namespace bewego
