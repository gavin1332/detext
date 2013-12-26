#ifndef COMMON_CORE_H_
#define COMMON_CORE_H_

#include <string>
#include <list>
#include <cmath>
#include <sstream>

#include <opencv2/core/core.hpp>

namespace dtxt {

enum Polarity {
  POS = 0,
  NEG = 1
};

class Const {
 public:
  static const uchar kFG = 255;
  static const uchar kMark = 100;
  static const uchar kBG = 0;

  static Polarity ParsePolarity(int code) {
    assert(code == 0 || code == 1);
    return (code == 0) ? POS : NEG;
  }

 private:
  Const();
  Const(const Const& const_);
  const Const& operator=(const Const& const_);

};

class Pixel {
 public:
  Pixel(const cv::Point& pos)
      : pos_(pos) {
  }

  const cv::Point& pos() const {
    return pos_;
  }

 private:
  const cv::Point pos_;

};

class Region {
 public:
  Region()
      : x1_(INT_MAX),
        y1_(INT_MAX),
        x2_(0),
        y2_(0) {
  }
  Region(int x1, int y1, int x2, int y2)
      : x1_(x1),
        y1_(y1),
        x2_(x2),
        y2_(y2) {
  }
  virtual ~Region() {
  }

  int x1() const {
    return x1_;
  }
  int y1() const {
    return y1_;
  }
  int x2() const {
    return x2_;
  }
  int y2() const {
    return y2_;
  }

  int Width() const {
    return x2_ - x1_ + 1;
  }
  int Height() const {
    return y2_ - y1_ + 1;
  }

  int Area() const {
    return Width() * Height();
  }

  double AspectRatio() const {
    return (double) Width() / Height();
  }

  double HRatio(const Region* cc) const {
    return (double) std::min(Height(), cc->Height())
        / std::max(Height(), cc->Height());
  }

  double WRatio(const Region* cc) const {
    return (double) std::min(Width(), cc->Width())
        / std::max(Width(), cc->Width());
  }

  cv::Rect ToCvRect() const {
    return cv::Rect(x1_, y1_, Width(), Height());
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "[" << x1_ << " " << y1_ << " " << x2_ << " " << y2_ << "]";
    return ss.str();
  }

 protected:
  int x1_, y1_, x2_, y2_;

};

typedef std::vector<Pixel*>::const_iterator PixelConstItr;
typedef std::vector<Pixel*>::iterator PixelItr;

class ConnComp : public Region {
 public:
  ConnComp()
      : Region(),
        valid_(true) {
  }
  ~ConnComp();

  bool valid() {
    return valid_;
  }
  void Invalidate() {
    valid_ = false;
  }

  int PixelNum() const {
    return pix_vec_.size();
  }

  double AreaRatio() const {
    return (double) PixelNum() / (Width() * Height());
  }

  const std::vector<Pixel*>& pix_vec() const {
    return pix_vec_;
  }

  cv::Point AnyPixelPos() const {
    return pix_vec_.front()->pos();
  }

  void AddPixel(Pixel* pixel) {
    pix_vec_.push_back(pixel);

    x1_ = std::min(x1_, pixel->pos().x);
    y1_ = std::min(y1_, pixel->pos().y);
    x2_ = std::max(x2_, pixel->pos().x);
    y2_ = std::max(y2_, pixel->pos().y);
  }

  void BuildMask(cv::Mat* mask) const;

  virtual void CheckValidation() = 0;

 protected:
  bool valid_;

  std::vector<Pixel*> pix_vec_;

};

typedef std::list<ConnComp*>::const_iterator CCConstItr;
typedef std::list<ConnComp*>::iterator CCItr;

class Char : public Region {
 public:
  Char(ConnComp* cc)
      : Region(cc->x1(), cc->y1(), cc->x2(), cc->y2()),
        value_(0) {
    cclist_.push_back(cc);
  }
  ~Char() {
  }

  ushort value() const {
    return value_;
  }
  void set_value(ushort value) {
    value_ = value;
  }

  int CCNum() const {
    return cclist_.size();
  }

  const std::list<ConnComp*>& cclist() {
    return cclist_;
  }

  bool WCovers(Region* rg) {
    return Width() >= rg->Width()
        && (rg->x1() >= x1() || abs(rg->x1() - x1()) < Width() / 6)
        && (rg->x2() <= x2() || abs(rg->x2() - x2()) < Width() / 6);
  }

  bool WCoveredBy(Region* rg) {
    return Width() <= rg->Width()
        && (rg->x1() <= x1() || abs(rg->x1() - x1()) < rg->Width() / 6)
        && (rg->x2() >= x2() || abs(rg->x2() - x2()) < rg->Width() / 6);
  }

 protected:
  ushort value_;
  // No need to release ConnComps
  std::list<ConnComp*> cclist_;

  Char();
};

// English char
class EnChar : public Char {
 public:
  enum Layout {
    MIDDLE_LINE = 1,
    TOP_MARGIN = 2,
    BOTTOM_MARGIN = 4
  };
  enum Style {
    STYLEa = MIDDLE_LINE,
    STYLEh = MIDDLE_LINE | TOP_MARGIN,
    STYLEy = MIDDLE_LINE | BOTTOM_MARGIN,
    STYLEf = MIDDLE_LINE | TOP_MARGIN | BOTTOM_MARGIN,
    UNDIFINED = 0
  };

  static Style ParseStyle(int code) {
    switch (code) {
      case STYLEa: return STYLEa;
      case STYLEh: return STYLEh;
      case STYLEy: return STYLEy;
      case STYLEf: return STYLEf;
      default: throw "Invalid Input";
    }
  }

  static char GetStyleName(Style style) {
    switch (style) {
      case STYLEa: return 'a';
      case STYLEh: return 'h';
      case STYLEy: return 'y';
      case STYLEf: return 'f';
      default: throw "Invalid Input";
    }
  }

  EnChar(ConnComp* cc)
      : Char(cc),
        style_(STYLEa) {
  }
  ~EnChar() {
  }

  Style style() const {
    return style_;
  }
  void set_style(Style style) {
    style_ = style;
  }
  void ExpandLayout(Layout layout) {
    style_ = EnChar::ParseStyle(style_ | layout);
  }

  ConnComp* GetCC() const {
    return cclist_.front();
  }

 private:
  Style style_;

  EnChar();

};

class CombChar : public Char {
 public:
  CombChar(ConnComp* cc)
      : Char(cc) {
  }

  // Only recieved width covered cc
  void AddConnComp(ConnComp* cc);

  void Swallow(Char* ch);

};

typedef std::list<Char*>::const_iterator CharConstItr;
typedef std::list<Char*>::iterator CharItr;

class TextLine : public Region {
 public:
  TextLine(int x1, int y1, int x2, int y2, const std::string& text = "")
      : Region(x1, y1, x2, y2),
        text_(text) {
  }

  const std::string& text() {
    return text_;
  }
  void set_text(const std::string& text) {
    text_ = text;
  }

  TextLine* NewZoomedTextLine(float factor) {
    return new TextLine(x1() * factor, y1() * factor, x2() * factor,
                        y2() * factor);
  }

 protected:
  std::string text_;

  TextLine() {
  }

};

typedef std::list<TextLine*>::const_iterator TextLineConstItr;
typedef std::list<TextLine*>::iterator TextLineItr;

class CharLine : public TextLine {
 public:
  CharLine(Char* ch)
      : TextLine(ch->x1(), ch->y1(), ch->x2(), ch->y2()),
        max_char_h_(ch->Height()),
        min_char_h_(ch->Height()),
        max_char_w_(ch->Width()) {
    clist_.push_back(ch);
  }
  ~CharLine();

  int CharNum() const {
    return clist_.size();
  }

  const std::list<Char*>& clist() const {
    return clist_;
  }

  int max_char_h() const {
    return max_char_h_;
  }

  int min_char_h() const {
    return min_char_h_;
  }

  int max_char_w() const {
    return max_char_w_;
  }

  int HeightStdDev() const;

  int MedianWidth() const;

  int MedianHeight() const;

  void CalcIntervalMeanAndStdDev(int* median, double* mean, double* stddev) const;

  void MedianY12(int* my1, int* my2) const;

  int CalcInterval(const Region* cc) const {
    int left = x1() - cc->x2();
    int right = cc->x1() - x2();
    return abs(left) < abs(right) ? left : right;
  }

  bool Contains(const Region* cc) const {
    return cc->x1() >= x1() && cc->y1() >= y1() && cc->x2() <= x2()
        && cc->y2() <= y2();
  }

  const Char* NeighborChar(const Region* cc) const {
    return (cc->x2() < x1() + Width() / 2) ? FrontChar() : BackChar();
  }

  const Char* FrontChar() const {
    return clist_.front();
  }
  const Char* BackChar() const {
    return clist_.back();
  }

  virtual void AddChar(Char* ch);

 protected:
  int max_char_h_;
  int min_char_h_;
  int max_char_w_;
  // Sorted by x2
  std::list<Char*> clist_;

  bool FollowListTail(Region* cc) {
    return (cc->x1() > clist_.back()->x1() && cc->x2() > clist_.back()->x2());
  }

};

class ConnCompLine : public CharLine {
 public:
  ConnCompLine(ConnComp* cc)
      : CharLine(new CombChar(cc)) {
  }
  ~ConnCompLine() {
  }

  // return true if CharNum() increased
  bool AddConnComp(ConnComp* cc);

 protected:
  // Check whether the chars before the forward iterator "fit" is width covered,
  // Swallow them if covered and value "true" is returned, false otherwise
  bool CheckWidthCoverage(CharItr fit);

};

}

#endif
