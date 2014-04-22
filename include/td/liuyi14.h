#ifndef TD_LIUYI14_H_
#define TD_LIUYI14_H_

#include "detext.h"

namespace dtxt {

class LiuYi14 : public TextDetector {
 public:
  LiuYi14() {
  }
  virtual ~LiuYi14() {
  }

  void Detect(const cv::Mat& img, std::list<TextLine*>* tllist);

 private:
  LiuYi14(const LiuYi14&);
  void operator=(const LiuYi14&);

  void GenRespMap(const cv::Mat& gray, cv::Mat* resp);

};

}

#endif
