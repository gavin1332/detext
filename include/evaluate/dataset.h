#ifndef EVALUATE_DATASET_H_
#define EVALUATE_DATASET_H_

#include <string>
#include <list>

#include "common/core.h"
#include "evaluate/evaluate.h"

namespace dtxt {

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

  virtual const std::string& test_data_dir() const {
    return base_dir_;
  }

  virtual const std::string& est_data_dir() const {
    return base_dir_;
  }

  void Report(double* precision, double* recall, double* f_measure) {
    evaluator_.Report(precision, recall, f_measure);
  }

  void IncEvaluate(const std::string& id, const std::list<TextLine*>& estlist);

  void SaveEstResult(const std::string& id,
                     const std::list<TextLine*>& estlist);

  void RetrieveTextLines(const std::string& id, std::list<TextLine*>* tllist);

  virtual void RetrieveImgIds(std::vector<std::string>* idlist) = 0;

 protected:
  std::string base_dir_;
  Evaluator evaluator_;

  virtual std::list<TextLine*>* RetrieveTgtList(const std::string& id) = 0;

  virtual void PostProcess(std::list<TextLine*>** tgtlist) = 0;

 private:
  static const std::string RESULT_POSTFIX;

};

}

#endif
