#ifndef EVALUATE_EVALUATE_H_
#define EVALUATE_EVALUATE_H_

#include <list>

#include "common/core.h"

class Evaluator {
 public:
  Evaluator()
      : precision_match_accum_(0),
        recall_match_accum_(0),
        E_count_(0),
        T_count_(0) {
  }
  virtual ~Evaluator() {
  }

  void Clear() {
    precision_match_accum_ = 0;
    recall_match_accum_ = 0;
    E_count_ = 0;
    T_count_ = 0;
  }

  void Report(float* precision, float* recall, float* f_measure);

  void RecordMatch(const std::list<dtxt::TextLine*>& E_list,
                   const std::list<dtxt::TextLine*>& T_list);

 private:
  static const float kAlpha;

  float precision_match_accum_;
  float recall_match_accum_;
  int E_count_;
  int T_count_;

  float CalcMatch(const dtxt::TextLine& input, const dtxt::TextLine& base);

  float CalcMatch(const dtxt::TextLine& input,
                  const std::list<dtxt::TextLine*>& base_list);

};

#endif
