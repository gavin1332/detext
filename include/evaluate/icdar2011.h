#ifndef EVALUATION_ICDAR2011_H_
#define EVALUATION_ICDAR2011_H_

#include <sstream>

#include "common/core.h"
#include "evaluate/dataset.h"

namespace dtxt {

class ICDAR2011 : public DataSet {
 public:
  ICDAR2011(const std::string& base_dir)
      : DataSet(base_dir) {
    test_data_dir_ = base_dir + "/test-textloc-gt/test-textloc-gt";
    est_data_dir_ = test_data_dir_;
  }
  virtual ~ICDAR2011() {
  }

  virtual const std::string& test_data_dir() const {
    return test_data_dir_;
  }

  virtual const std::string& est_data_dir() const {
    return est_data_dir_;
  }

  std::list<dtxt::TextLine*>* RetrieveTgtList(const std::string& img_path);

  void PostProcess(std::list<dtxt::TextLine*>** tgtlist);

  void RetrieveImgIds(std::vector<std::string>* idlist);

 private:
  std::string test_data_dir_;
  std::string est_data_dir_;
  std::string BuildTextLineFilePath(const std::string& img_path);

};

}

#endif
