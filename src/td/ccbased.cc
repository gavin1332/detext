#include "td/ccbased.h"

#include <stack>
#include <fstream>
#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>

#include "common/core.h"
#include "common/test.h"

using namespace std;
using namespace cv;
using namespace dtxt;

void AttrConnComp::CheckValidation() {
  if (AspectRatio() < 0.45 && AreaRatio() > 0.7) {
    upright_ = true;
  } else if (Height() < 9 || Height() > 355 || PixelNum() < 35) {
    // TODO: PixelNum() < 35 is too small
    Invalidate();
  } else if (AspectRatio() > 2 || AspectRatio() < 0.1) {
    Invalidate();
  } else if (AreaRatio() < 0.17 || AreaRatio() > 0.85) {
    Invalidate();
  } else if (AspectRatio() > 1.1 && AreaRatio() > 0.7) {
    Invalidate();
  }
}

void AttrEnCharLine::ExpandLayout(EnChar::Layout layout) {
  if ((style_ & layout) != 0) {
    return;
  }
  CharItr end = clist_.end();
  for (CharItr itr = clist_.begin(); itr != end; ++itr) {
    static_cast<EnChar*>(*itr)->ExpandLayout(layout);
  }
  if (layout == EnChar::TOP_MARGIN) {
    est_h_height_ = est_a_height_;
    est_f_height_ = est_y_height_;
    est_a_height_ = 0;
    est_y_height_ = 0;
  } else {
    est_y_height_ = est_a_height_;
    est_f_height_ = est_h_height_;
    est_a_height_ = 0;
    est_h_height_ = 0;
  }
  style_ = EnChar::ParseStyle(style_ | layout);
  TestUtils::Log("expands char line style to", style_);
}

void AttrEnCharLine::UpdateEstHeight(EnChar* ch) {
  int *updated_value = NULL;
  switch (ch->style()) {
    case EnChar::STYLEa:
      updated_value = &est_a_height_;
      break;
    case EnChar::STYLEh:
      updated_value = &est_h_height_;
      break;
    case EnChar::STYLEy:
      updated_value = &est_y_height_;
      break;
    case EnChar::STYLEf:
      updated_value = &est_f_height_;
      break;
    default:
      throw "Invalid Input";
  }
  *updated_value =
      *updated_value == 0 ?
          ch->Height() :
          (*updated_value * CharNum() + ch->Height()) / (CharNum() + 1);
}

void AttrEnCharLine::AddChar(EnChar* ch) {
  if (ch->style() != (style_ & ch->style())) {
    style_ = EnChar::ParseStyle(style_ | ch->style());
    TestUtils::Log("expands char line style to", style_);
  }
  UpdateEstHeight(ch);
  CharLine::AddChar(ch);
}

bool AttrEnCharLine::AreMostUpright() const {
  int upright_num = 0;
  for (CharConstItr itr = clist().begin(); itr != clist().end(); ++itr) {
    if ((*itr)->CCNum() == 1
        && static_cast<AttrConnComp*>(static_cast<EnChar*>(*itr)->GetCC())
            ->upright()) {
      ++upright_num;
    }
  }
  return upright_num > CharNum() * 3 / 4;
}

bool AttrEnCharLine::CheckValidation() {
  if (!valid()) {
    return false;
  } else if (CharNum() <= 2) {
    TestUtils::Log("too less ccs in one tr");
    Invalidate();
    return false;
  } else if (AspectRatio() < 1.2) {
    TestUtils::Log("too small aspect ratio");
    Invalidate();
    return false;
  } else if (Width() < CharNum() * max_char_w_ * 0.375) {
    TestUtils::Log("too small width");
    Invalidate();
    return false;
  } else if (AreMostUpright()) {
    TestUtils::Log("most are upright");
    Invalidate();
    return false;
  }
  return true;
}

Polarity ConnCompBased::polarity_;

void ConnCompBased::GroupConnComp(const Mat& resp_mask, list<ConnComp*>* cclist,
                                  list<TextLine*>* tllist) {
  extern bool SHOW_GROUP_STEP_;

//  cclist->sort(CompareY1);

  CCItr begin = cclist->begin();
  CCItr end = cclist->end();
  Mat map;
  BuildCCMap(&map, resp_mask.size(), begin, end);
  if (SHOW_GROUP_STEP_) {
    TestUtils::ShowImage(map);
  }
  bool show_steps = SHOW_GROUP_STEP_;
  vector<CCItr> candidates;
  CCItr ccbase = begin;
  while (ccbase != end) {
    if (!(*ccbase)->valid()) {
      ++ccbase;
      continue;
    }
    TestUtils::Log("new char line begins");
    AttrEnCharLine* tl = new AttrEnCharLine(new EnChar(*ccbase), polarity_);
    (*ccbase)->Invalidate();
    bool char_added = true;

    CCItr outer = ccbase;
    while (char_added && outer != end) {
      char_added = false;
      candidates.clear();
      FindIntervalSortedCandidates(tl, outer, end, &candidates);
      if (candidates.empty()) {
        break;
      }

      vector<CCItr>::iterator inner = candidates.begin();
      for (; inner != candidates.end(); ++inner) {
        if (show_steps) {
          show_steps &= !HilightCharLine(map, tl);
        }
        if (show_steps) {
          show_steps &= !HilightClAndCC(map, tl, **inner);
        }

        EnChar* ch = new EnChar(**inner);
        if (!CheckLayout(tl, ch)) {
          TestUtils::Log("conflicted char style");

          // TODO: Refine
          if ((tl->style() != EnChar::STYLEa && tl->CharNum() <= 3)
              || (tl->style() == EnChar::STYLEf && tl->CharNum() <= 4)) {
            tl->Invalidate();
          }
          continue;
        }

        tl->AddChar(ch);
        TestUtils::Log("Add the ch with style", ch->style());
        (**inner)->Invalidate();
        char_added = true;

        if (show_steps) {
          show_steps &= !HilightCharLine(map, tl, Scalar(255, 0, 255));
        }
      }
    }

    // TODO: 添加根据char line style过滤的逻辑
    if (!tl->CheckValidation()) {
      delete tl;
    } else {
      TestUtils::Log("accept a char line");
      PrintTextLineStyle(tl);
      tllist->push_back(tl);
    }

    for (++ccbase; ccbase != end && !(*ccbase)->valid(); ++ccbase) {
    }
  }

  if (show_steps) {
    DispRects(resp_mask, *tllist, Scalar(0, 255, 0));
  }
}

bool ConnCompBased::CheckHeight(AttrEnCharLine* cl, EnChar* ch) {
  return true;
  int est_height = 0;
  switch (ch->style()) {
    case EnChar::STYLEa:
      est_height = cl->est_a_height();
      break;
    case EnChar::STYLEh:
      if (ch->Height() < cl->est_a_height()) {
        return false;
      }
      est_height = cl->est_h_height();
      break;
    case EnChar::STYLEy:
      if (ch->Height() < cl->est_a_height()) {
        return false;
      }
      est_height = cl->est_y_height();
      break;
    case EnChar::STYLEf:
      if (ch->Height() < cl->est_a_height()) {
        return false;
      }
      est_height = cl->est_f_height();
      break;
    default:
      throw "Invalid Input";
  }
  return (est_height == 0 || StrictMatchHeight(est_height, ch->Height()));
}

bool ConnCompBased::CheckLayout(AttrEnCharLine* cl, EnChar* ch) {
  const EnChar* neib = static_cast<const EnChar*>(cl->NeighborChar(ch));
  const double kDiffRatioThres = 1.1;
  int dy1 = ch->y1() - neib->y1();
  int dy2 = ch->y2() - neib->y2();
  bool matchy1 = MatchY1(neib, ch);
  bool matchy2 = MatchY2(neib, ch);

  if (matchy1 && matchy2) {
    if (StrictMatchHeight(neib->Height(), ch->Height())) {
      ch->set_style(neib->style());
      return true;
    }
  }

  if (!matchy1 && !matchy2) {
    const int kDiffHThres = min(ch->Height(), cl->min_char_h())
        * ((cl->CharNum() == 1 && ch->x2() > cl->x2()) ? 0.8 : 1.1);
    if (abs(dy1) < kDiffHThres && abs(dy2) < kDiffHThres) {
      if (dy1 < 0 && dy2 > 0 && neib->style() == EnChar::STYLEa) {
        ch->set_style(EnChar::STYLEf);
        return CheckHeight(cl, ch);
      }
      if (dy1 > 0 && dy2 > 0) {
        if (neib->style() == EnChar::STYLEh) {
          ch->set_style(EnChar::STYLEy);
          return CheckHeight(cl, ch);
        }
        if (neib->style() == EnChar::STYLEa) {
          ch->set_style(EnChar::STYLEy);
          cl->ExpandLayout(EnChar::TOP_MARGIN);
          return CheckHeight(cl, ch);
        }
      }
      if (dy1 < 0 && dy2 < 0) {
        if (neib->style() == EnChar::STYLEy) {
          ch->set_style(EnChar::STYLEh);
          return CheckHeight(cl, ch);
        }
        if (neib->style() == EnChar::STYLEa) {
          cl->ExpandLayout(EnChar::BOTTOM_MARGIN);
          ch->set_style(EnChar::STYLEh);
          return CheckHeight(cl, ch);
        }
      }
      if (dy1 > 0 && dy2 < 0) {
        if (neib->style() == EnChar::STYLEf) {
          ch->set_style(EnChar::STYLEa);
          return CheckHeight(cl, ch);
        }
        if (neib->style() == EnChar::STYLEa) {
          cl->ExpandLayout(EnChar::TOP_MARGIN);
          cl->ExpandLayout(EnChar::BOTTOM_MARGIN);
          ch->set_style(EnChar::STYLEa);
          return CheckHeight(cl, ch);
        }
      }
    }
    return false;
  }

  bool less_dy1 = abs(neib->y1() - ch->y1()) < abs(neib->y2() - ch->y2());
  if (less_dy1) {
    if (abs(dy2) < min(ch->Height(), neib->Height()) * kDiffRatioThres) {
      if (dy2 < 0) {
        if (neib->style() == EnChar::STYLEa
            || neib->style() == EnChar::STYLEh) {
          ch->set_style(neib->style());
          cl->ExpandLayout(EnChar::BOTTOM_MARGIN);
          return CheckHeight(cl, ch);
        }
        if (neib->style() == EnChar::STYLEy
            || neib->style() == EnChar::STYLEf) {
          ch->set_style(
              EnChar::ParseStyle(neib->style() & ~EnChar::BOTTOM_MARGIN));
          return CheckHeight(cl, ch);
        }
      } else if (neib->style() == EnChar::STYLEa
          || neib->style() == EnChar::STYLEh) {
        ch->set_style(neib->style());
        ch->ExpandLayout(EnChar::BOTTOM_MARGIN);
        return CheckHeight(cl, ch);
      }
    }
  } else {
    if (abs(dy1) < min(ch->Height(), neib->Height()) * kDiffRatioThres) {
      if (dy1 > 0) {
        if (neib->style() == EnChar::STYLEa
            || neib->style() == EnChar::STYLEy) {
          ch->set_style(neib->style());
          cl->ExpandLayout(EnChar::TOP_MARGIN);
          return CheckHeight(cl, ch);
        }
        if (neib->style() == EnChar::STYLEh
            || neib->style() == EnChar::STYLEf) {
          ch->set_style(
              EnChar::ParseStyle(neib->style() & ~EnChar::TOP_MARGIN));
          return CheckHeight(cl, ch);
        }
      } else if (neib->style() == EnChar::STYLEa
          || neib->style() == EnChar::STYLEy) {
        ch->set_style(neib->style());
        ch->ExpandLayout(EnChar::TOP_MARGIN);
        return CheckHeight(cl, ch);
      }
    }
  }
  return false;
}

void ConnCompBased::PrintTextLineStyle(AttrEnCharLine* tl) {
  CharConstItr end = tl->clist().end();
  cout << "char line style: ";
  for (CharConstItr itr = tl->clist().begin(); itr != end; ++itr) {
    cout << EnChar::GetStyleName(static_cast<EnChar*>(*itr)->style());
  }
  cout << endl;
}

void ConnCompBased::FindIntervalSortedCandidates(AttrEnCharLine* tl,
                                                 CCItr begin, CCItr end,
                                                 vector<CCItr>* candidates) {
  for (; begin != end && InSearchStripe(tl, *begin); ++begin) {
    if ((*begin)->valid() && HasProperInterval(tl, *begin)
        && HasProperWH(tl, *begin)
        && HasAcceptableVerticalShift(tl->NeighborChar(*begin), *begin)) {
      candidates->push_back(begin);
    }
  }
  sort(candidates->begin(), candidates->end(), [&](CCItr it1, CCItr it2) {
    int dist1 =
    (*it1)->x2() > tl->x2() ?
    (*it1)->x1() - tl->x2() : tl->x1() - (*it1)->x2();
    int dist2 =
    (*it2)->x2() > tl->x2() ?
    (*it2)->x1() - tl->x2() : tl->x1() - (*it2)->x2();
    return dist1 < dist2;
  });
}

void ConnCompBased::BuildCCMap(Mat* map, Size map_size, CCItr begin,
                               CCItr end) {
  *map = Mat::zeros(map_size, CV_8UC1);
  list<ConnComp*>::reverse_iterator rit(end), rend(begin);
  uchar value;
  for (; rit != rend; ++rit) {
    AttrConnComp* cc = static_cast<AttrConnComp*>(*rit);
    if (cc->valid()) {
      value = Const::kFG;
    } else {
      value = Const::kMark;
    }
    PixelConstItr pit = cc->pix_vec().begin();
    PixelConstItr pend = cc->pix_vec().end();
    for (; pit != pend; ++pit) {
      map->at<uchar>((*pit)->pos()) = value;
    }
  }
}

bool ConnCompBased::IsOverlapped(TextLine* tl1, TextLine* tl2) const {
  int inner_x1, inner_y1, inner_x2, inner_y2;
  inner_x1 = max(tl1->x1(), tl2->x1());
  inner_y1 = max(tl1->y1(), tl2->y1());
  inner_x2 = min(tl1->x2(), tl2->x2());
  inner_y2 = min(tl1->y2(), tl2->y2());
  if (inner_x1 >= inner_x2 || inner_y1 >= inner_y2) {
    return false;
  }

  int inner_area = (inner_x2 - inner_x1 + 1) * (inner_y2 - inner_y1 + 1);
  double area_ratio = (double) inner_area / min(tl1->Area(), tl2->Area());
  return area_ratio > 0.72;
}

void ConnCompBased::OverlapAnalyse(list<TextLine*>* tllist) {
  TextLineItr it = tllist->begin();
  while (it != tllist->end()) {
    AttrEnCharLine* itl = static_cast<AttrEnCharLine*>(*it);
    bool it_erased = false;
//    double ihstddev = itl->HeightStdDev();
    TextLineItr jt = it;
    ++jt;
    while (jt != tllist->end()) {
      AttrEnCharLine* jtl = static_cast<AttrEnCharLine*>(*jt);
      if (itl->polarity() == jtl->polarity() || !IsOverlapped(itl, jtl)) {
        ++jt;
        continue;
      }
//      double jhstddev = jtl->HeightStdDev();
//      if (ihstddev < jhstddev) {
//        jt = tllist->erase(jt);
//      } else {
//        it = tllist->erase(it);
//        it_erased = true;
//        break;
//      }
      // TODO: Fix equal condition
      if (itl->CharNum() >= jtl->CharNum()) {
        jt = tllist->erase(jt);
      } else {
        it = tllist->erase(it);
        it_erased = true;
        break;
      }
    }
    if (!it_erased) {
      ++it;
    }
  }
}

CharLine* ConnCompBased::CreateCharLine(CharConstItr begin,
                                              CharConstItr end) {
  assert(begin != end);
  CharLine* cl = new CharLine(*begin);
  for (++begin; begin != end; ++begin) {
    cl->AddChar(new EnChar(static_cast<EnChar*>(*begin)->GetCC()));
  }
  return cl;
}

void ConnCompBased::SplitCharLine(list<TextLine*>* tllist) {
  list<TextLine*>::iterator itr = tllist->begin();
  while (itr != tllist->end()) {
    CharLine* cl = static_cast<CharLine*>(*itr);
    if (cl->CharNum() < 4) {
      ++itr;
      continue;
    }

    int median;
    double mean, stddev;
    // OPTIMIZE
    cl->CalcIntervalMeanAndStdDev(&median, &mean, &stddev);
    const double kIntervalThres = mean + 2 * stddev;
    TestUtils::Log("split width threshold", kIntervalThres);

    const list<Char*>& clist = cl->clist();
    CharConstItr begin = clist.begin();
    CharConstItr end = clist.end();
    CharConstItr ait = begin;

    CharConstItr bit = ait;
    for (++bit; bit != end; ait = bit++) {
      int interval = (*bit)->x1() - (*ait)->x2();
      if (interval > kIntervalThres) {
        tllist->insert(itr, CreateCharLine(begin, bit));
        begin = bit;
      }
    }

    if (begin != clist.begin()) {
      if (begin != end) {
        tllist->insert(itr, CreateCharLine(begin, end));
      }
      cout << "split " << (cl)->ToString() << endl;
//      delete cl;
      itr = tllist->erase(itr);
    } else {
      ++itr;
    }
  }

  itr = tllist->begin();
  while (itr != tllist->end()) {
    CharLine* cl = static_cast<CharLine*>(*itr);
    if (cl->CharNum() == 1) {
      TestUtils::Log("Remove single char", (*itr)->ToString());
//      delete *itr;
      itr = tllist->erase(itr);
    } else {
      ++itr;
    }
  }
}

void ConnCompBased::FillConnComp(Mat* rgb, const ConnComp* cc, Scalar bgr) {
  for (Pixel* pix : cc->pix_vec()) {
    Vec3b* ptr = rgb->ptr<Vec3b>(pix->pos().y, pix->pos().x);
    (*ptr)[0] = bgr[0];
    (*ptr)[1] = bgr[1];
    (*ptr)[2] = bgr[2];
  }
}

bool ConnCompBased::HilightCharLine(const Mat& gray, const CharLine* cl,
                                    Scalar bgr) {
  Mat color;
  cvtColor(gray, color, CV_GRAY2BGR);
  for (Char* ch : cl->clist()) {
    for (ConnComp* cc : ch->cclist()) {
      FillConnComp(&color, cc, bgr);
    }
  }
  return TestUtils::ShowImage(color);
}

bool ConnCompBased::HilightClAndCC(const Mat& gray, const CharLine* cl,
                                   const ConnComp* cc, Scalar bgr) {
  Mat color;
  cvtColor(gray, color, CV_GRAY2BGR);
  for (Char* ch : cl->clist()) {
    for (ConnComp* acc : ch->cclist()) {
      FillConnComp(&color, acc, bgr);
    }
  }
  FillConnComp(&color, cc, Scalar(255, 255, 255) - bgr);
  return TestUtils::ShowImage(color);
}
