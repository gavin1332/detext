#ifndef EVALUATE_DATASET_H_
#define EVALUATE_DATASET_H_

#include <string>
#include <list>

#include "common/core.h"

class DataSet {
 public:
  DataSet(const std::string& base_dir)
      : base_dir_(base_dir) {
  }
  virtual ~DataSet() {
  }

  const std::string& base_dir() {
    return base_dir_;
  }

  virtual const std::list<dtxt::TextLine*>& RetrieveTList(
      const std::string& img_path) = 0;

 protected:
  std::string base_dir_;

};

#endif
