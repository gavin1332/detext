#ifndef EVALUATE_ICDAR_H_
#define EVALUATE_ICDAR_H_

#include <string>
#include <list>
#include <map>

#include "evaluate/dataset.h"
#include "common/core.h"

class ICDAR : public DataSet {
 public:
  ICDAR(const std::string& base_dir)
      : DataSet(base_dir) {
  }
  virtual ~ICDAR();

  const std::string& test_data_dir() {
    return test_data_dir_;
  }

 protected:
  std::string test_data_dir_;

  std::map<std::string, std::list<dtxt::TextLine*>*> T_list_map_;

};

#endif
