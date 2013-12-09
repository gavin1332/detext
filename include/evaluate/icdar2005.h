#ifndef EVALUATE_ICDAR2005_H_
#define EVALUATE_ICDAR2005_H_

#include <string>
#include <list>
#include <map>

#include "common/core.h"
#include "evaluate/icdar.h"

class ICDAR2005 : public ICDAR {
 public:
  ICDAR2005(const std::string& base_dir)
      : ICDAR(base_dir) {
    test_data_dir_ = base_dir + "/SceneTrialTest";
    InitTListMap();
  }
  virtual ~ICDAR2005() {
  }

  const std::list<dtxt::TextLine*>& RetrieveTList(const std::string& img_path);

 private:
  void InitTListMap();

};

#endif  // EVALUATE_ICDAR_H_
