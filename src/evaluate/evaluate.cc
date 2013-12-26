#include "evaluate/evaluate.h"

#include <cmath>

#include "common/utils.h"

using namespace std;
using namespace dtxt;

const float Evaluator::kAlpha = 0.5f;

void Evaluator::Report(double* precision, double* recall, double* f_measure) {
  *precision = (est_count_ == 0) ? 0 : precision_match_accum_ / est_count_;
  *recall = (tgt_count_ == 0) ? 1 : recall_match_accum_ / tgt_count_;
  if (MathUtils::IsAbsErr0(*precision) || MathUtils::IsAbsErr0(*recall)) {
    *f_measure = 0;
  } else {
    *f_measure = 1 / (kAlpha / *precision + (1 - kAlpha) / *recall);
  }
}

void Evaluator::RecordMatch(const list<TextLine*>& estlist,
                            const list<TextLine*>& tgtlist) {
  for (TextLine* tl : estlist) {
    precision_match_accum_ += CalcMatch(*tl, tgtlist);
  }
  est_count_ += estlist.size();
  for (TextLine* tl : tgtlist) {
    recall_match_accum_ += CalcMatch(*tl, estlist);
  }
  tgt_count_ += tgtlist.size();
}

double Evaluator::CalcMatch(const TextLine& input, const TextLine& base) {
  int inner_x1 = max(input.x1(), base.x1());
  int inner_y1 = max(input.y1(), base.y1());
  int inner_x2 = min(input.x2(), base.x2());
  int inner_y2 = min(input.y2(), base.y2());

  if (inner_x1 >= inner_x2 || inner_y1 >= inner_y2) {
    return 0;
  }

  int outer_x1 = min(input.x1(), base.x1());
  int outer_y1 = min(input.y1(), base.y1());
  int outer_x2 = max(input.x2(), base.x2());
  int outer_y2 = max(input.y2(), base.y2());

  double inner_area = (inner_x2 - inner_x1 + 1) * (inner_y2 - inner_y1 + 1);
  double outer_area = (outer_x2 - outer_x1 + 1) * (outer_y2 - outer_y1 + 1);

  return inner_area / outer_area;
}

double Evaluator::CalcMatch(const TextLine& input,
                            const list<TextLine*>& base_list) {
  double match = 0;
  list<TextLine*>::const_iterator it = base_list.begin();
  for (; it != base_list.end(); ++it) {
    double temp = CalcMatch(input, **it);
    match = (temp > match) ? temp : match;
  }

  return match;
}
