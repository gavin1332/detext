#ifndef TD_LIUYI13_H_
#define	TD_LIUYI13_H_

#include <list>

#include <opencv2/core/core.hpp>

#include "ccbased.h"

namespace dtxt {

class LiuYi13 : public ConnCompBased {
 public:
  LiuYi13() {
  }
  virtual ~LiuYi13() {
  }

  void Detect(const cv::Mat& img, std::list<TextLine*>* trlist);

 private:
  void ReleaseCCList(std::list<ConnComp*>* cclist);

  // the cclist is in y1 increasing order
  void ParseResp(const cv::Mat& gray, const cv::Mat* resp, cv::Mat* mask,
                 std::list<ConnComp*>* cclist);

  void CheckCCValidation(std::list<ConnComp*>* cclist);

  void GenRespMap(const cv::Mat& gray, cv::Mat* resp);

  LiuYi13(const LiuYi13&);
  void operator=(const LiuYi13&);

};

}

#endif

