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

#include "socket.h"

#include <unistd.h>

using namespace std;
using namespace bewego::util;

TcpClient::TcpClient() {
  sock = -1;
  port = 0;
  address = "";
}

// Connect to a host on a certain port number
bool TcpClient::Connect(string address, int port) {
  // create socket if it is not already created
  if (sock == -1) {
    // Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
      perror("Could not create socket");
    }

    cout << "Socket created\n";
  } else { /* OK , nothing */
  }

  // setup address structure
  if (inet_addr(address.c_str()) == -1) {
    struct hostent *he;
    struct in_addr **addr_list;

    // resolve the hostname, its not an ip address
    if ((he = gethostbyname(address.c_str())) == NULL) {
      // gethostbyname failed
      herror("gethostbyname");
      cout << "Failed to resolve hostname\n";
      return false;
    }

    // Cast the h_addr_list to in_addr , since h_addr_list also has the ip
    // address in long format only
    addr_list = (struct in_addr **)he->h_addr_list;

    for (int i = 0; addr_list[i] != NULL; i++) {
      // strcpy(ip , inet_ntoa(*addr_list[i]) );
      server.sin_addr = *addr_list[i];

      cout << address << " resolved to " << inet_ntoa(*addr_list[i]) << endl;

      break;
    }
  }

  // plain ip address
  else {
    server.sin_addr.s_addr = inet_addr(address.c_str());
  }

  server.sin_family = AF_INET;
  server.sin_port = htons(port);

  // Connect to remote server
  if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
    perror("connect failed. Error");
    return false;
  }

  cout << "Connected\n";
  return true;
}

// Send data to the connected host
bool TcpClient::SendMessage(string data) {
  uint32_t msg_length = strlen(data.c_str());
  cout << "msg_length : " << msg_length << endl;
  if (send(sock, (char *)(&msg_length), 4, 0) < 0) {
    perror("Send failed : ");
    return false;
  }

  return SendData(data);
}

// Send data to the connected host
bool TcpClient::SendData(string data) {
  if (verbose_) {
    cout << "Sending data...";
    cout << data;
    cout << "\n";
  }
  // Send some data
  if (send(sock, data.c_str(), strlen(data.c_str()), 0) < 0) {
    perror("Send failed : ");
    return false;
  }

  cout << "Data send\n";

  return true;
}

// Receive data from the connected host
string TcpClient::Receive(int size = 512) {
  char buffer[size];
  string reply;

  // Receive a reply from the server
  if (recv(sock, buffer, sizeof(buffer), 0) < 0) {
    puts("recv failed");
    return NULL;
  }

  reply = buffer;
  response_data = reply;

  return reply;
}

void TcpClient::Close() { close(sock); }
