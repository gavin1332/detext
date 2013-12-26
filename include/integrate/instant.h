#ifndef INTEGRATE_INSTANT_H_
#define INTEGRATE_INSTANT_H_

#include "offline.h"
#include "evaluate/dataset.h"
#include "td/detext.h"

namespace dtxt {

class InstantProcess : public OfflineProcess {
 public:
  InstantProcess(DataSet* dataset, TextDetector* detector)
      : OfflineProcess(dataset),
        detector_(detector) {
  }

  void Run();

 private:
  TextDetector* detector_;

  void PrintProgress(int curr, int total);

  void Report(double exec_time);

  bool ShowPreComputedResult(const cv::Mat& img, const std::string& filename);

  void ZoomTextLine(std::list<TextLine*>** estlist, double factor);

};

}

#endif
