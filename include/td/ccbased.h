#ifndef TD_REGIONBASED_H_
#define TD_REGIONBASED_H_

#include <string>
#include <list>
#include <cmath>

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
        upright_(false) {
  }
  virtual ~AttrConnComp() {
  }

  uchar median_gray() const {
    return median_gray_;
  }

  bool upright() const {
    return upright_;
  }

  void CheckValidation();

 private:
  static const int kCrossVarNum = 3;

  uchar median_gray_;
  bool upright_;

  static bool PixGrayCompare(const Pixel* p1, const Pixel* p2) {
    return static_cast<const AttrPix*>(p1)->gray_value()
        < static_cast<const AttrPix*>(p2)->gray_value();
  }

  void CalcMedianGray();

};

class AttrEnChar : public EnChar {
 public:
  AttrEnChar(ConnComp* cc)
      : EnChar(cc) {
  }

  uchar median_gray() const {
    return static_cast<const AttrConnComp*>(GetCC())->median_gray();
  }

};

class AttrEnCharLine : public CharLine {
 public:
  AttrEnCharLine(Polarity polarity)
      : CharLine(),
        polarity_(polarity),
        median_gray_mean_(0),
        style_(EnChar::STYLEa) {
  }

  Polarity polarity() const {
    return polarity_;
  }

  uchar median_gray_mean() const {
    return median_gray_mean_;
  }

  EnChar::Style style() const {
    return style_;
  }
  void UpdateLayout(EnChar::Layout layout);

  void AddChar(EnChar* ch);

 protected:
  const Polarity polarity_;
  uchar median_gray_mean_;
  EnChar::Style style_;

};

class ConnCompBased : public TextDetector {
 protected:
  static Polarity polarity_;

  ConnCompBased() {
  }

  void set_polarity(Polarity polarity) {
    polarity_ = polarity;
  }

  void GroupConnComp(const cv::Mat& resp_mask, std::list<ConnComp*>* cclist,
                     std::list<TextLine*>* trlist);

  void OverlapAnalyse(std::list<TextLine*>* trlist);

  void SplitCharLine(std::list<TextLine*>* cllist);

  void BuildCCMap(cv::Mat* map, cv::Size map_size, CCItr begin, CCItr end);

 private:
  ConnCompBased(const ConnCompBased&);
  void operator=(const ConnCompBased&);

  static bool CompareY1(ConnComp* cc1, ConnComp* cc2) {
    return cc1->y1() < cc2->y1();
  }

  bool IsInSearchScope(const CharLine* tr, const ConnComp* ch) const {
    return ch->y1() < tr->y2() - tr->max_char_h() / 3;
  }

  bool CheckInOneLine(CharLine* tr, Char* ch);

  bool MostAreUpright(const CharLine* tl) const;

  bool MatchY1(const Region* r1, const Region* r2) const {
    return abs(r1->y1() - r2->y1()) < std::min(r1->Height(), r2->Height()) / 4.7;
  }

  bool MatchY2(const Region* r1, const Region* r2) const {
    return abs(r1->y2() - r2->y2()) < std::min(r1->Height(), r2->Height()) / 4.7;
  }

  bool IsOverlapped(TextLine* tl1, TextLine* tl2) const;

  void FindSortedCandidates(AttrEnCharLine* tl, CCItr begin, CCItr end,
                            std::vector<CCItr>* candidates);

  void PrintTextLineStyle(AttrEnCharLine* tl);

  CharLine* CreateAttrEnCharLine(CharConstItr begin, CharConstItr end);

};

}

#endif
