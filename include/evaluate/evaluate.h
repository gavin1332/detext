#ifndef EVALUATE_EVALUATE_H_
#define EVALUATE_EVALUATE_H_

#include <list>

#include "common/core.h"

namespace dtxt {

class Evaluator {
 public:
  Evaluator()
      : precision_match_accum_(0.0),
        recall_match_accum_(0.0),
        est_count_(0),
        tgt_count_(0) {
  }
  virtual ~Evaluator() {
  }

  void Clear() {
    precision_match_accum_ = 0.0;
    recall_match_accum_ = 0.0;
    est_count_ = 0;
    tgt_count_ = 0;
  }

  void Report(double* precision, double* recall, double* f_measure);

  void IncRecord(const std::list<TextLine*>& estlist,
                 const std::list<TextLine*>& tgtlist);

 private:
  static const double kAlpha;

  double precision_match_accum_;
  double recall_match_accum_;
  int est_count_;
  int tgt_count_;

  double CalcMatch(const TextLine& input, const TextLine& base);

  double CalcMatch(const TextLine& input,
                   const std::list<TextLine*>& base_list);

};

}

#endif
