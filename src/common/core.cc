#include "common/core.h"

#include <stack>
#include <list>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>

#include "common/utils.h"

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

void CombChar::AddConnComp(ConnComp* cc) {
  assert(WCovers(cc));
  y1_ = min(y1_, cc->y1());
  y2_ = max(y2_, cc->y2());
  cclist_.push_back(cc);
}

void CombChar::Swallow(Char* ch) {
  assert(WCovers(ch));
  CCConstItr end = ch->cclist().end();
  for (CCConstItr itr = ch->cclist().begin(); itr != end; ++itr) {
    AddConnComp(*itr);
  }
}

CharLine::~CharLine() {
  CharItr itr = clist_.begin();
  for (; itr != clist_.end(); ++itr) {
    delete *itr;
  }
  clist_.clear();
}

int CharLine::HeightStdDev() const {
  vector<int> hvec;
  CharConstItr itr = clist_.begin();
  for (; itr != clist_.end(); ++itr) {
    hvec.push_back((*itr)->Height());
  }
  double mean, stddev;
  MathUtils::MeanAndStdDev(hvec, &mean, &stddev);
  return stddev;
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

void CharLine::CalcIntervalMeanAndStdDev(int* median, double* mean, double* stddev) const {
  vector<int> int_vec;
  CharConstItr end = clist_.end();
  CharConstItr ait = clist_.begin();
  CharConstItr bit = ait;
  for (++bit; bit != end; ait = bit++) {
    int interval = max(0, (*bit)->x1() - (*ait)->x2());
    int_vec.push_back(interval);
  }
  size_t n = (int_vec.size() - 1) / 2;
  nth_element(int_vec.begin(), int_vec.begin() + n, int_vec.end());
  *median = int_vec[n];

  MathUtils::MeanAndStdDev(int_vec, mean, stddev);
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

void CharLine::AddChar(Char* ch) {
  x1_ = min(x1_, ch->x1());
  y1_ = min(y1_, ch->y1());
  x2_ = max(x2_, ch->x2());
  y2_ = max(y2_, ch->y2());

  if (max_char_h_ < ch->Height()) {
    max_char_h_ = ch->Height();
  }
  if (min_char_h_ > ch->Height()) {
    min_char_h_ = ch->Height();
  }
  if (max_char_w_ < ch->Width()) {
    max_char_w_ = ch->Width();
  }

  if (clist_.empty() || FollowListTail(ch)) {
    clist_.push_back(ch);
    return;
  }

  CharItr begin = clist_.begin();
  CharItr end = clist_.end();
  CharItr itr = begin;
  while (itr != end && ch->x2() > (*itr)->x2()) {
    ++itr;
  }
  itr = clist_.insert(itr, ch);
}

// TODO: part of CharLine's parameters are not updated
bool ConnCompLine::AddConnComp(ConnComp* cc) {
  x1_ = min(x1_, cc->x1());
  y1_ = min(y1_, cc->y1());
  x2_ = max(x2_, cc->x2());
  y2_ = max(y2_, cc->y2());

  if (clist_.empty() || FollowListTail(cc)) {
    clist_.push_back(new CombChar(cc));
    return true;
  }

  CharItr begin = clist_.begin();
  CharItr end = clist_.end();
  CharItr itr = begin;
  while (itr != end && cc->x2() > (*itr)->x2()) {
    ++itr;
  }
  if (itr == end) {
    itr = clist_.insert(itr, new CombChar(cc));
    return !CheckWidthCoverage(itr);
  }
  if ((*itr)->WCovers(cc)) {
    static_cast<CombChar*>(*itr)->AddConnComp(cc);
    return false;
  }
  if ((*itr)->WCoveredBy(cc)) {
    itr = clist_.insert(++itr, new CombChar(cc));
    CheckWidthCoverage(itr);
    return false;
  }
  itr = clist_.insert(itr, new CombChar(cc));
  return true;
}

bool ConnCompLine::CheckWidthCoverage(CharItr fit) {
  bool covered = false;
  CharItr itr = fit;
  while (itr-- != clist_.begin() && (*fit)->WCoveredBy(*itr)) {
    static_cast<CombChar*>(*fit)->Swallow(*itr);
    delete *itr;
    itr = clist_.erase(itr);
    covered = true;
  }
  return covered;
}
