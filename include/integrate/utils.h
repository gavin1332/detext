#ifndef INTEGRATE_UTILS_H_
#define INTEGRATE_UTILS_H_

#include <list>

#include <opencv2/core/core.hpp>

#include "common/core.h"

namespace dtxt {

class IntegrateUtils {
 public:
  static bool ShowTextLines(const std::string& title, const cv::Mat& gray_rgb,
                            const std::list<dtxt::TextLine*>& tllist,
                            cv::Scalar bgr = cv::Scalar(255, 255, 255),
                            int thickness = 2);

  static void ReleaseList(std::list<dtxt::TextLine*>* tllist);

 private:
  IntegrateUtils();
  ~IntegrateUtils();

};

}

#endif
