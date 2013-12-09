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

  std::vector<Pixel*>& pix_vec() {
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
  Char(ConnComp* cc, const char& ch = '\0')
     : Region(cc->x1(), cc->y1(), cc->x2(), cc->y2()),
       ch_(ch) {
    cclist_.push_back(cc);
  }
  ~Char() {}

  const std::list<ConnComp*>& cclist() {
    return cclist_;
  }

  bool WidthCovers(ConnComp* cc) {
    return cc->x1() >= x1() && cc->x2() <= x2();
  }

  bool CoverWidth(Char* ch) {
    return ch->x1() >= x1() && ch->x2() <= x2();
  }

  bool WidthCoveredBy(ConnComp* cc) {
    return cc->x1() <= x1() && cc->x2() >= x2();
  }

  // Only recieved width covered cc
  void AddConnComp(ConnComp* cc);

  void Swallow(Char* ch) {
    assert(CoverWidth(ch));
    cclist_.insert(cclist_.end(), ch->cclist().begin(), ch->cclist().end());
  }

 private:
  char ch_;
  // No need to release ConnComps
  std::list<ConnComp*> cclist_;

  Char();
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

  TextLine* TuneRect(float factor) {
    return new TextLine(x1() * factor, y1() * factor, x2() * factor,
                        y2() * factor);
  }

 protected:
  std::string text_;

  TextLine() {
  }

};

class CharLine : public TextLine {
 public:
  CharLine()
      : TextLine(INT_MAX, INT_MAX, 0, 0) {
  }
  ~CharLine();

  int CharNum() const {
    return clist_.size();
  }

  const std::list<Char*>& clist() const {
    return clist_;
  }

  int MedianWidth() const;

  int MedianHeight() const;

  int MedianInterval() const;

  void MedianY12(int* my1, int* my2) const;

  int CalcWidthInterval(const ConnComp* cc) const {
    if (cc->x2() < x1() || cc->x1() > x2()) {
      return std::min(abs(x1() - cc->x2()), abs(x2() - cc->x1()));
    } else {
      return 0;
    }
  }

  bool Contains(const ConnComp* cc) const {
    return cc->x1() >= x1() && cc->y1() >= y1() && cc->x2() <= x2()
        && cc->y2() <= y2();
  }

  const Char* NeighborChar(const ConnComp* cc) const {
    return (cc->x2() < x1() + Width() / 2) ? FrontChar() : BackChar();
  }

  const Char* FrontChar() const {
    return clist_.front();
  }
  const Char* BackChar() const {
    return clist_.back();
  }

  // return true if CharNum() increased
  virtual bool AddConnComp(ConnComp* cc);

 protected:
  std::list<Char*> clist_;

  bool FollowListTail(ConnComp* cc) {
    return (cc->x1() > clist_.back()->x1() && cc->x2() > clist_.back()->x2());
  }

  // Check whether the chars before the forward iterator "fit" is width covered,
  // Swallow them if covered and value "true" is returned, false otherwise
  bool CheckWidthCoverage(CharItr fit);

};

}

#endif
