#include "common/core.h"

#include <stack>
#include <list>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace dtxt;

void ConnComp::BuildMask(Mat* mask) const {
  *mask = Mat::zeros(Height(), Width(), CV_8UC1);
  Point shift(-x1_, -y1_);
  PixelConstItr it = pix_vec_.begin();
  PixelConstItr end = pix_vec_.end();
  for (; it != end; ++it) {
    mask->at<uchar>((*it)->pos() + shift) = Const::kFG;
  }
}

ConnComp::~ConnComp() {
  PixelItr itr = pix_vec_.begin();
  for (; itr != pix_vec_.end(); ++itr) {
    delete (*itr);
  }
  pix_vec_.clear();
}

void Char::AddConnComp(ConnComp* cc) {
  assert(cc->x1() >= x1() && cc->x2() <= x2());
  y1_ = min(y1_, cc->y1());
  y2_ = max(y2_, cc->y2());
}

CharLine::~CharLine() {
  CharItr itr = clist_.begin();
  for (; itr != clist_.end(); ++itr) {
    delete *itr;
  }
  clist_.clear();
}

bool CharLine::AddConnComp(ConnComp* cc) {
  x1_ = min(x1_, cc->x1());
  y1_ = min(y1_, cc->y1());
  x2_ = max(x2_, cc->x2());
  y2_ = max(y2_, cc->y2());

  if (clist_.empty() || FollowListTail(cc)) {
    clist_.push_back(new Char(cc));
    return true;
  }

  CharItr begin = clist_.begin();
  CharItr end = clist_.end();
  CharItr itr = begin;
  while (itr != end && cc->x2() > (*itr)->x2()) {
    ++itr;
  }
  if (itr == end) {
    itr = clist_.insert(itr, new Char(cc));
    return !CheckWidthCoverage(itr);
  }
  if ((*itr)->WidthCovers(cc)) {
    (*itr)->AddConnComp(cc);
    return false;
  }
  if ((*itr)->WidthCoveredBy(cc)) {
    itr = clist_.insert(++itr, new Char(cc));
    CheckWidthCoverage(itr);
    return false;
  }
  itr = clist_.insert(itr, new Char(cc));
  return true;
}

bool CharLine::CheckWidthCoverage(CharItr fit) {
  bool covered = false;
  CharItr erasable;
  Char* curr = *fit;
  list<Char*>::reverse_iterator rit(fit);
  for (; rit != clist_.rend() && curr->CoverWidth(*rit); ++rit) {
    curr->Swallow(*rit);
    erasable = --rit.base();
    delete *erasable;
    clist_.erase(erasable);
    covered = true;
  }
  return covered;
}

int CharLine::MedianWidth() const {
  vector<int> wvec;
  CharConstItr itr = clist_.begin();
  for (; itr != clist_.end(); ++itr) {
    wvec.push_back((*itr)->Width());
  }
  size_t n = (wvec.size() - 1) / 2;
  nth_element(wvec.begin(), wvec.begin() + n, wvec.end());
  return wvec[n];
}

int CharLine::MedianHeight() const {
  vector<int> hvec;
  CharConstItr itr = clist_.begin();
  for (; itr != clist_.end(); ++itr) {
    hvec.push_back((*itr)->Height());
  }
  size_t n = (hvec.size() - 1) / 2;
  nth_element(hvec.begin(), hvec.begin() + n, hvec.end());
  return hvec[n];
}

int CharLine::MedianInterval() const {
  vector<int> int_vec;
  CharConstItr end = clist_.end();
  CharConstItr ait = clist_.begin();
  CharConstItr bit = ait;
  for (; ait != end; ait = bit) {
    if (bit == end) {
      break;
    }
    int interval = (*bit)->x1() - (*ait)->x2();
    int_vec.push_back(interval);
  }
  size_t n = (int_vec.size() - 1) / 2;
  nth_element(int_vec.begin(), int_vec.begin() + n, int_vec.end());
  return int_vec[n];
}

void CharLine::MedianY12(int* my1, int* my2) const {
  vector<int> y1vec;
  vector<int> y2vec;
  CharConstItr itr = clist_.begin();
  for (; itr != clist_.end(); ++itr) {
    y1vec.push_back((*itr)->y1());
    y2vec.push_back((*itr)->y2());
  }
  size_t n = (y1vec.size() - 1) / 2;
  nth_element(y1vec.begin(), y1vec.begin() + n, y1vec.end());
  *my1 = y1vec[n];
  nth_element(y2vec.begin(), y2vec.begin() + n, y2vec.end());
  *my2 = y2vec[n];
}
