#ifndef EVALUATE_ICDAR2005_H_
#define EVALUATE_ICDAR2005_H_

#include <string>
#include <list>
#include <map>

#include "common/core.h"
#include "evaluate/dataset.h"

namespace dtxt {

class ICDAR2005 : public DataSet {
 public:
  ICDAR2005(const std::string& base_dir)
      : DataSet(base_dir) {
    test_data_dir_ = base_dir + "/SceneTrialTest";
    est_data_dir_ = test_data_dir_;
    InitTListMap();
  }
  ~ICDAR2005();

  const std::string& test_data_dir() const {
    return test_data_dir_;
  }

  const std::string& est_data_dir() const {
    return est_data_dir_;
  }

  std::list<dtxt::TextLine*>* RetrieveTgtList(const std::string& img_path);

  void PostProcess(std::list<dtxt::TextLine*>** tgtlist) {
  }

  void RetrieveImgIds(std::vector<std::string>* idlist);

 private:
  std::string test_data_dir_;
  std::string est_data_dir_;
  std::map<std::string, std::list<dtxt::TextLine*>*> T_list_map_;

  void InitTListMap();

};

}

#endif
