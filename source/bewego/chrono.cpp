// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include "chrono.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <iostream>

using std::cout;
using std::endl;

namespace bewego {

#define NCHRONO 20
static int print_flag = true;
static int counter = -1;
static double t_init;
static struct timeval beg_tod[NCHRONO];
static int counter_tod = -1;
static int print_flag_tod = -1;
//
int ChronoOn(void) {
  if (++counter_tod >= NCHRONO) {
    fprintf(stderr, "ChronoTimeOfDayOn : Warning: Chrono stack overflow\n");
    return (--counter_tod);
  }

  gettimeofday(&beg_tod[counter_tod], NULL);
  return (counter_tod);
}

int ChronoOff(void) {
  if (counter_tod == -1) {
    fprintf(stderr, "ChronoTimeOfDayOff : Warning: Chrono stack underflow\n");
    return (-1);
  }
  return (counter_tod--);
}

int ChronoTimes(double *tu, double *ts) {
  if ((counter_tod == -1) || (counter_tod >= NCHRONO)) {
    *tu = 0.0;
    *ts = 0.0;
    return (-1);
  }

  timeval tim;
  gettimeofday(&tim, NULL);

  *tu = double(tim.tv_usec - beg_tod[counter_tod].tv_usec);
  *ts = double(tim.tv_sec - beg_tod[counter_tod].tv_sec);

  return (counter);
}

int ChronoTimesSeconds(double *ts) {
  if ((counter_tod == -1) || (counter_tod >= NCHRONO)) {
    *ts = 0.0;
    return (-1);
  }

  timeval tim;
  gettimeofday(&tim, NULL);

  double tu;

  tu = double(tim.tv_usec - beg_tod[counter_tod].tv_usec);
  *ts = double(tim.tv_sec - beg_tod[counter_tod].tv_sec);

  *ts = *ts + tu / 1000000.0;

  return (counter);
}

void ChronoPrint(const char *msg) {
  double ts;
  int i;
  if (print_flag) {
    for (i = 0; i < counter_tod + 1; i++) {
      ChronoTimesSeconds(&ts);
      fprintf(stdout, "Clock = %6.3f sec : ", ts);
      fprintf(stdout, "  ");
      fprintf(stdout, "[%d] %s\n", counter_tod, msg);
    }
  }
}

}  // namespace bewego