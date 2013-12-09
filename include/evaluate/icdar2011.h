#ifndef EVALUATION_ICDAR2011_H_
#define EVALUATION_ICDAR2011_H_

#include <sstream>

#include "common/core.h"
#include "evaluate/icdar.h"

class ICDAR2011 : public ICDAR {
 public:
  ICDAR2011(const std::string& base_dir)
      : ICDAR(base_dir) {
    test_data_dir_ = base_dir + "/test-textloc-gt/test-textloc-gt";
  }
  virtual ~ICDAR2011() {
  }

  const std::list<dtxt::TextLine*>& RetrieveTList(const std::string& img_path);

 private:
  std::string BuildTextLineFilePath(const std::string& img_path);

};

#endif
