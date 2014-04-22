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
        upright_(false) {
  }
  virtual ~AttrConnComp() {
  }

  bool upright() const {
    return upright_;
  }

  void CheckValidation();

 private:
  static const int kCrossVarNum = 3;

  bool upright_;

  static bool PixGrayCompare(const Pixel* p1, const Pixel* p2) {
    return static_cast<const AttrPix*>(p1)->gray_value()
        < static_cast<const AttrPix*>(p2)->gray_value();
  }

  void CalcMedianGray();

};

class AttrEnCharLine : public CharLine {
 public:
  AttrEnCharLine(EnChar* ch, Polarity polarity)
      : CharLine(ch),
        polarity_(polarity),
        valid_(true),
        style_(EnChar::STYLEa),
        est_a_height_(ch->Height()),
        est_h_height_(0),
        est_y_height_(0),
        est_f_height_(0) {
  }

  Polarity polarity() const {
    return polarity_;
  }

  bool valid() const {
    return valid_;
  }
  void Invalidate() {
    valid_ = false;
  }

  int est_a_height() const {
    return est_a_height_;
  }
  int est_h_height() const {
    return est_h_height_;
  }
  int est_y_height() const {
    return est_y_height_;
  }
  int est_f_height() const {
    return est_f_height_;
  }

  EnChar::Style style() const {
    return style_;
  }
  void ExpandLayout(EnChar::Layout layout);

  void AddChar(EnChar* ch);

  bool CheckValidation();

 private:
  const Polarity polarity_;
  bool valid_;

  EnChar::Style style_;
  int est_a_height_;
  int est_h_height_;
  int est_y_height_;
  int est_f_height_;

  bool AreMostUpright() const;

  void UpdateEstHeight(EnChar* ch);

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
                     std::list<TextLine*>* tllist);

  void OverlapAnalyse(std::list<TextLine*>* tllist);

  void SplitCharLine(std::list<TextLine*>* cllist);

  void BuildCCMap(cv::Mat* map, cv::Size map_size, CCItr begin, CCItr end);

 private:
  ConnCompBased(const ConnCompBased&);
  void operator=(const ConnCompBased&);

  static bool CompareY1(ConnComp* cc1, ConnComp* cc2) {
    return cc1->y1() < cc2->y1();
  }

  bool InSearchStripe(const CharLine* tl, const ConnComp* ch) const {
    return ch->y1() < tl->y2() - tl->max_char_h() / 3;
  }

  bool HasProperInterval(const CharLine* tl, const ConnComp* ch) const {
    int interval = tl->CalcInterval(ch);
    if (interval >= 0) {
      return interval < std::min(tl->min_char_h(), ch->Height()) / 1.5;
    } else {
      return -interval < tl->max_char_w() / 1.9;
    }
  }

  // TODO: not applicable for icdar2011 325.jpg
  bool HasProperWH(const CharLine* tl, const ConnComp* ch) const {
    return ch->Width() < tl->min_char_h() * 2.4
        && ch->Height() < tl->min_char_h() * 3.0
        && ch->Height() > tl->max_char_h() / 3.0;
  }

  bool HasAcceptableVerticalShift(const Char* ch, const Region* rg) const {
    int ady1 = abs(ch->y1() - rg->y1());
    int ady2 = abs(ch->y2() - rg->y2());
    return MatchY1(ch, rg) || MatchY2(ch, rg)
        || (double) std::max(ady1, ady2) / std::min(ady1, ady2) < 1.8;
  }

  bool CheckLayout(AttrEnCharLine* cl, EnChar* ch);

  bool StrictMatchHeight(int h1, int h2) const {
    return abs(h1 - h2) < std::min(h1, h2) / 6.5;
  }

  bool CheckHeight(AttrEnCharLine* cl, EnChar* ch);

  bool MatchY1(const Region* r1, const Region* r2) const {
    return abs(r1->y1() - r2->y1()) < std::min(r1->Height(), r2->Height()) / 4.7;
  }

  bool MatchY2(const Region* r1, const Region* r2) const {
    return abs(r1->y2() - r2->y2()) < std::min(r1->Height(), r2->Height()) / 4.7;
  }

  bool IsOverlapped(TextLine* tl1, TextLine* tl2) const;

  void FindIntervalSortedCandidates(AttrEnCharLine* cl, CCItr begin, CCItr end,
                                    std::vector<CCItr>* candidates);

  void PrintTextLineStyle(AttrEnCharLine* tl);

  CharLine* CreateCharLine(CharConstItr begin, CharConstItr end);

  bool HilightCharLine(const cv::Mat& gray, const CharLine* cl, cv::Scalar bgr =
                           cv::Scalar(0, 0, 255));

  bool HilightClAndCC(const cv::Mat& gray, const CharLine* cl,
                      const ConnComp* cc,
                      cv::Scalar bgr = cv::Scalar(0, 0, 255));

  void FillConnComp(cv::Mat* rgb, const ConnComp* cc, cv::Scalar bgr);

};

}

#endif
