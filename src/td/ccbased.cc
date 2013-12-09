#include "td/ccbased.h"

#include <stack>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

#include "common/core.h"
#include "common/test.h"

using namespace std;
using namespace cv;
using namespace dtxt;

// TODO: 如果一个联通区域包含多个有效的小联通区域，则判定为无效
void AttrConnComp::CheckValidation() {
  if (!valid()) {
    return;
  }
  if (AspectRatio() < 0.45 && AreaRatio() > 0.7) {
    upright_ = true;
    return;
  }
  // TODO: PixelNum() < 50 is too small
  if (Height() < 10 || Height() > 350 || PixelNum() < 50) {
    Invalidate();
    return;
  }
  if (AspectRatio() > 2 || AspectRatio() < 0.1) {
    Invalidate();
    return;
  }
  if (AreaRatio() < 0.2 || AreaRatio() > 0.85) {
    Invalidate();
    return;
  }
  if (AspectRatio() > 1.1 && AreaRatio() > 0.7) {
    Invalidate();
    return;
  }
  CalcProperties();
//  for (int i = 0; i < kCrossVarNum; ++i) {
//    if (hcross[i] > 5 || vcross[i] > 5) {
//      Invalidate();
//      break;
//    }
//  }
}

void AttrConnComp::CalcProperties() {
  CalcMedianGray();
  CalcGrayStdDev();
  CalcMaskProperties();
}

void AttrConnComp::CalcMedianGray() {
  int n = (PixelNum() - 1) / 2;
  PixelItr begin = pix_vec().begin();
  nth_element(begin, begin + n, pix_vec().end(), PixGrayCompare);
  median_gray_ = static_cast<AttrPix*>(*(begin + n))->gray_value();
}

void AttrConnComp::CalcGrayStdDev() {
  double mean = 0;
  PixelConstItr itr = pix_vec().begin();
  PixelConstItr end = pix_vec().end();
  for (; itr != end; ++itr) {
    mean += static_cast<AttrPix*>(*itr)->gray_value();
  }
  mean /= pix_vec().size();

  itr = pix_vec().begin();
  for (; itr != end; ++itr) {
    double factor = static_cast<AttrPix*>(*itr)->gray_value() - mean;
    gray_stddev_ += factor * factor;
  }
  gray_stddev_ = sqrt(gray_stddev_ / pix_vec().size());
}

void AttrConnComp::CalcMaskProperties() {
  Mat mask;
  BuildBorderedMask(&mask);

  perimeter_ = 0;
  memset(hcross, 0, kCrossVarNum * sizeof(int));
  memset(vcross, 0, kCrossVarNum * sizeof(int));

  int ori_width = Width();
  int ori_height = Height();
  int step = mask.step1();
  for (int y = 0; y < ori_height; ++y) {
    uchar* ptr = mask.ptr(y + 1) + 1;
    for (int x = 0; x < ori_width; ++x, ++ptr) {
      if (*ptr == 255) {
        if (*(ptr - step - 1) == 0 || *(ptr - step) == 0
            || *(ptr - step + 1) == 0 || *(ptr - 1) == 0 || *(ptr + 1) == 0
            || *(ptr + step - 1) == 0 || *(ptr + step) == 0
            || *(ptr + step + 1) == 0) {
          ++perimeter_;
        }

        if (*(ptr - 1) == 0) {
          if (y == ori_height / 4) {
            ++hcross[0];
          }
          if (y == ori_height / 2) {
            ++hcross[1];
          }
          if (y == ori_height * 3 / 4) {
            ++hcross[2];
          }
        }

        if (*(ptr - step) == 0) {
          if (x == ori_width / 4) {
            ++vcross[0];
          }
          if (x == ori_width / 2) {
            ++vcross[1];
          }
          if (x == ori_width * 3 / 4) {
            ++vcross[2];
          }
        }
      }
    }
  }
}

void AttrConnComp::BuildBorderedMask(Mat* mask) {
  *mask = Mat::zeros(Height() + 2, Width() + 2, CV_8UC1);
  Point shift(1 - x1(), 1 - y1());
  PixelConstItr itr = pix_vec().begin();
  PixelConstItr end = pix_vec().end();
  for (; itr != end; ++itr) {
    mask->at<uchar>((*itr)->pos() + shift) = 255;
  }
}

bool AttrCharLine::AddConnComp(ConnComp* cc) {
  bool new_char_added = CharLine::AddConnComp(cc);
  median_gray_mean_ = static_cast<uchar>((median_gray_mean_ * CharNum()
      + static_cast<AttrConnComp*>(cc)->median_gray()) / (CharNum() + 1));
  if (cc->Height() > max_cc_h_) {
    max_cc_h_ = cc->Height();
  }
  return new_char_added;
}

void ConnCompBased::GroupConnComp(const Mat& gray, list<ConnComp*>* cclist,
                                  list<TextLine*>* tllist) {
  extern bool SHOW_GROUP_STEP_;

  cclist->sort(CompareY1);

  CCItr begin = cclist->begin();
  CCItr end = cclist->end();
  Mat map;
  BuildCCMap(&map, gray.size(), begin, end);
  if (SHOW_GROUP_STEP_) {
    TestUtils::ShowImage(map);
  }

  const uchar kLooseThres = 35;
  bool show_steps = SHOW_GROUP_STEP_;
  CCItr ccbase = begin;
  while (ccbase != end) {
    if (!(*ccbase)->valid()) {
      ++ccbase;
      continue;
    }
    AttrCharLine* tl = new AttrCharLine();
    tl->AddConnComp(*ccbase);
    (*ccbase)->Invalidate();

    CCItr itr = ccbase;
    for (++itr; itr != end; ++itr) {
      AttrConnComp* acc = static_cast<AttrConnComp*>(*itr);
      if (!IsInSearchScope(tl, acc)) {
        break;
      }
      if (!acc->valid()) {
        continue;
      }

      if (show_steps) {
        TestUtils::ShowRect(map, tl->ToCvRect(), Scalar(0, 0, 255), 2);
        show_steps &= !TestUtils::ShowRect(map, acc->ToCvRect(),
                                           Scalar(0, 255, 0));
      }

      const int kDistThres = acc->Height() / 2;
      if (tl->CalcWidthInterval(acc) > kDistThres) {
        TestUtils::Log("too wide interval");
        continue;
      }

      bool add_cc = false;
      uchar median_gray_diff = abs(acc->median_gray() - tl->median_gray_mean());
//      TestUtils::Log("median_gray_diff", (int) median_gray_diff);
      if (IsInOneLine(tl, acc)) {
        add_cc = true;
        TestUtils::Log("they are in one line and add the cc");
      } else if (tl->Contains(acc) && median_gray_diff < kLooseThres) {
        add_cc = true;
        TestUtils::Log("the base contains cc and add the cc");
      }
      if (!add_cc) {
        continue;
      }

      tl->AddConnComp(acc);
      acc->Invalidate();
      itr = ccbase;
      if (show_steps) {
        show_steps &= !TestUtils::ShowRect(map, tl->ToCvRect(),
                                           Scalar(0, 100, 255), 2);
      }
    }

    if (tl->CharNum() > 2) {
      tllist->push_back(tl);
    } else {
      delete tl;
    }

    for (++ccbase; ccbase != end && !(*ccbase)->valid(); ++ccbase) {
    }
  }
}

// TODO: check
bool ConnCompBased::IsInOneLine(const AttrCharLine* tl,
                                const AttrConnComp* cc) {
  int my1, my2;
  tl->MedianY12(&my1, &my2);
  int mh = my2 - my1;

  bool top_or_bottom_match = min(abs(cc->y1() - my1), abs(cc->y2() - my2))
      < (my2 - my1) / 3 || abs(cc->y1() - tl->y1()) < (my2 - my1) / 3
      || abs(cc->y2() - tl->y2()) < (my2 - my1) / 3;
  double h_max_h_ratio = (double) std::min(cc->Height(), mh)
      / std::max(cc->Height(), mh);
  if (h_max_h_ratio > 0.5 && top_or_bottom_match) {
    return true;
  }
  if (MatchY(tl->NeighborChar(cc), cc)) {
    return true;
  }
  if (tl->CharNum() > 3) {
    int median_h = tl->MedianHeight();
    double h_median_h_ratio = (double) std::min(cc->Height(), median_h)
        / std::max(cc->Height(), median_h);
    return h_median_h_ratio > 0.75 && top_or_bottom_match;
  }
  return false;
}

void ConnCompBased::BuildCCMap(Mat* map, Size map_size, CCItr begin,
                               CCItr end) {
  *map = Mat::zeros(map_size, CV_8UC1);
  list<ConnComp*>::reverse_iterator rit(end), rend(begin);
  uchar value;
  for (; rit != rend; ++rit) {
    AttrConnComp* cc = static_cast<AttrConnComp*>(*rit);
    if (cc->valid()) {
      value = Const::kFG;
    } else {
      value = Const::kMark;
    }
    PixelConstItr pit = cc->pix_vec().begin();
    PixelConstItr pend = cc->pix_vec().end();
    for (; pit != pend; ++pit) {
      map->at<uchar>((*pit)->pos()) = value;
    }
  }
}
