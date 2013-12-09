#ifndef TD_DETEXT_H_
#define	TD_DETEXT_H_

#include <list>

#include <opencv2/core/core.hpp>

#include "common/core.h"

namespace dtxt {

class TextDetector {
 public:
  virtual ~TextDetector() {
  }
  virtual void Detect(const cv::Mat& img,
                      std::list<dtxt::TextLine*>* trlist) = 0;

 protected:
  TextDetector() {
  }
  void DispRects(const cv::Mat& gray, const std::list<dtxt::TextLine*>& trlist,
                 cv::Scalar color = cv::Scalar(255, 255, 255));

 private:
  TextDetector(const TextDetector&);
  void operator=(const TextDetector&);

};

}

#endif

