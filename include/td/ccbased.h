#ifndef TD_REGIONBASED_H_
#define TD_REGIONBASED_H_

#include <string>
#include <list>
#include <cmath>
#include <sstream>

#include <opencv2/core/core.hpp>

#include "common/core.h"
#include "td/detext.h"

namespace dtxt {

class AttrPix : public Pixel {
 public:
  AttrPix(cv::Point pos, uchar gray_value, double resp_value)
      : Pixel(pos),
        gray_value_(gray_value),
        resp_value_(resp_value) {
  }

  uchar gray_value() const {
    return gray_value_;
  }

  double resp_value() const {
    return resp_value_;
  }

 private:
  uchar gray_value_;
  double resp_value_;
};

class AttrConnComp : public ConnComp {
 public:
  AttrConnComp()
      : ConnComp(),
        median_gray_(0),
        gray_stddev_(0.0),
        upright_(false),
        perimeter_(0) {
  }
  virtual ~AttrConnComp() {
  }

  uchar median_gray() {
    return median_gray_;
  }

  bool upright() {
    return upright_;
  }

  void CalcProperties();

  void CheckValidation();

 private:
  static const int kCrossVarNum = 3;

  uchar median_gray_;
  double gray_stddev_;
  bool upright_;
  int perimeter_;
  int hcross[kCrossVarNum];
  int vcross[kCrossVarNum];

  static bool PixGrayCompare(const Pixel* p1, const Pixel* p2) {
    return static_cast<const AttrPix*>(p1)->gray_value()
        < static_cast<const AttrPix*>(p2)->gray_value();
  }

  void CalcMedianGray();

  void CalcGrayStdDev();

  void CalcMaskProperties();

  void BuildBorderedMask(cv::Mat* mask);

  double gray_stddev() {
    return gray_stddev_;
  }

  int TotalHCross() const {
    return hcross[0] + hcross[1] + hcross[2];
  }
  int TotalVCross() const {
    return vcross[0] + vcross[1] + vcross[2];
  }

};

class AttrCharLine : public CharLine {
 public:
  AttrCharLine()
      : CharLine(),
        max_cc_h_(0),
        median_gray_mean_(0) {
  }

  int max_cc_h() const {
    return max_cc_h_;
  }

  uchar median_gray_mean() {
    return median_gray_mean_;
  }

  bool AddConnComp(ConnComp* r);

 private:
  int max_cc_h_;
  uchar median_gray_mean_;

};

class ConnCompBased : public TextDetector {
 protected:
  ConnCompBased() {
  }

  void GroupConnComp(const cv::Mat& gray, std::list<ConnComp*>* cclist,
                     std::list<TextLine*>* trlist);

  void BuildCCMap(cv::Mat* map, cv::Size map_size, CCItr begin, CCItr end);

 private:
  ConnCompBased(const ConnCompBased&);
  void operator=(const ConnCompBased&);

  static bool CompareY1(ConnComp* cc1, ConnComp* cc2) {
    return cc1->y1() < cc2->y1();
  }

  bool IsInSearchScope(const AttrCharLine* tr, const AttrConnComp* cc) {
    return cc->y1() < tr->y2() - tr->max_cc_h() / 3;
  }

  bool IsInOneLine(const AttrCharLine* tr, const AttrConnComp* region);

  bool MatchY(const Char* ch, const ConnComp* cc) const {
    return (abs(ch->y1() - cc->y1()) < ch->Height() / 6
        && abs(ch->y2() - cc->y2()) < ch->Height() / 6)
        || (abs(ch->y1() - cc->y1()) < ch->Height() / 4 && ch->HRatio(cc) > 0.85);
  }
};

}

#endif
