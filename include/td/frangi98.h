#ifndef TD_FRANGI98_H_
#define	TD_FRANGI98_H_

#include <list>

#include <opencv2/core/core.hpp>

#include "ccbased.h"

namespace dtxt {

class Frangi98 : public ConnCompBased {
 public:
  Frangi98() {
  }
  virtual ~Frangi98() {
  }

  void Detect(const cv::Mat& img, std::list<TextLine*>* tllist);

 private:
  static const double EPSILON;

  void Handle(const cv::Mat& gray, std::list<TextLine*>* tllist);

  void genRespMap(const cv::Mat& fgray, double sigma, cv::Mat* resp,
                  cv::Mat* mask);

  void GroupRegion(const cv::Mat& binary, std::vector<Region*>* region_vec,
                   std::list<TextLine*>* tllist);

  Frangi98(const Frangi98&);
  void operator=(const Frangi98&);

};

}

#endif

