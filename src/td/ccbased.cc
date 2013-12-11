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

// TODO: 如果一个联通区域包含多个有效的小联通区域，则判定为无效
void AttrConnComp::CheckValidation() {
  if (AspectRatio() < 0.45 && AreaRatio() > 0.7) {
    upright_ = true;
  } else if (Height() < 10 || Height() > 355 || PixelNum() < 50) {
    // TODO: PixelNum() < 50 is too small
    Invalidate();
  } else if (AspectRatio() > 2 || AspectRatio() < 0.1) {
    Invalidate();
  } else if (AreaRatio() < 0.17 || AreaRatio() > 0.85) {
    Invalidate();
  } else if (AspectRatio() > 1.1 && AreaRatio() > 0.7) {
    Invalidate();
  }
  if (valid()) {
    CalcMedianGray();
  }
}

void AttrConnComp::CalcMedianGray() {
  int n = (PixelNum() - 1) / 2;
  PixelItr begin = pix_vec().begin();
  nth_element(begin, begin + n, pix_vec().end(), PixGrayCompare);
  median_gray_ = static_cast<AttrPix*>(*(begin + n))->gray_value();
}

void AttrEnCharLine::UpdateLayout(EnChar::Layout layout) {
  if ((style_ & layout) != 0) {
    return;
  }
  CharItr end = clist_.end();
  for (CharItr itr = clist_.begin(); itr != end; ++itr) {
    static_cast<EnChar*>(*itr)->UpdateLayout(layout);
  }
  style_ = EnChar::ParseStyle(style_ | layout);
}

void AttrEnCharLine::AddChar(EnChar* ch) {
  const int char_num = CharNum();
  median_gray_mean_ = static_cast<uchar>((median_gray_mean_ * char_num
      + static_cast<AttrEnChar*>(ch)->median_gray()) / (char_num + 1));
  if (ch->style() != (style_ & ch->style())) {
    style_ = EnChar::ParseStyle(style_ | ch->style());
    TestUtils::Log("update style", style_);
  }
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
  } else if (Width() < CharNum() * max_char_w_ * 0.75) {
    TestUtils::Log("too small width");
    Invalidate();
    return false;
  } else if (AreMostUpright ()) {
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
    AttrEnCharLine* tl = new AttrEnCharLine(polarity_);
    tl->AddChar(new AttrEnChar(*ccbase));
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
          show_steps &= !TestUtils::ShowRect(map, tl->ToCvRect(),
                                             Scalar(0, 0, 255), 2);
        }
        if (show_steps) {
          show_steps &= !TestUtils::ShowRect(map, (**inner)->ToCvRect(),
                                             Scalar(0, 255, 0), 2);
        }

        AttrEnChar* ch = new AttrEnChar(**inner);
        if (!CheckStyle(tl, ch)) {
          delete ch;
          TestUtils::Log("conflicted char style");

          // TODO: Refine
          if ((tl->style() != EnChar::STYLEa && tl->CharNum() <= 3)
              || (tl->style() == EnChar::STYLEf && tl->CharNum() <= 4)) {
            tl->Invalidate();
            break;
          }
          continue;
        }

        TestUtils::Log("Add the ch");
        tl->AddChar(ch);
        (**inner)->Invalidate();
        char_added = true;

        if (show_steps) {
          show_steps &= !TestUtils::ShowRect(map, tl->ToCvRect(),
                                             Scalar(255, 255, 0), 2);
        }
      }
    }

    // TODO: 添加根据Text line style过滤的逻辑
    if (!tl->CheckValidation()) {
      delete tl;
    }else if (tl->x1() < 5 || tl->x2() > resp_mask.cols - 5 || tl->y1() < 10
        || tl->y2() > resp_mask.rows - 10) {
      // TODO: Check the necessary of this condition
      TestUtils::Log("ignore lines on the edges");
      delete tl;
    } else {
      TestUtils::Log("accept a text line");
      PrintTextLineStyle(tl);
      tllist->push_back(tl);
    }

    for (++ccbase; ccbase != end && !(*ccbase)->valid(); ++ccbase) {
    }
  }

  if (show_steps) {
    DispRects(resp_mask, *tllist, Scalar(0, 0, 255));
  }
}

void ConnCompBased::PrintTextLineStyle(AttrEnCharLine* tl) {
  CharConstItr end = tl->clist().end();
  cout << "text line style: ";
  for (CharConstItr itr = tl->clist().begin(); itr != end; ++itr) {
    cout << EnChar::GetStyleName(static_cast<EnChar*>(*itr)->style());
  }
  cout << endl;
}

void ConnCompBased::FindIntervalSortedCandidates(AttrEnCharLine* tl,
                                                 CCItr begin, CCItr end,
                                                 vector<CCItr>* candidates) {
  for (; begin != end; ++begin) {
    if (!(*begin)->valid()) {
      continue;
    }
    if (!IsInSearchScope(tl, *begin)) {
      break;
    }
    if (tl->CalcInterval(*begin) > (*begin)->Height() / 1.5) {
      continue;
    }
    candidates->push_back(begin);
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

bool ConnCompBased::CheckStyle(CharLine* cl, Char* ch) {
  AttrEnCharLine* ch_line = static_cast<AttrEnCharLine*>(cl);
  const EnChar* neib = static_cast<const EnChar*>(ch_line->NeighborChar(ch));
  EnChar* ench = static_cast<EnChar*>(ch);
  if (MatchY1(neib, ench) && MatchY2(neib, ench)) {
    ench->set_style(neib->style());
    return true;
  }

  const double kDiffRatioThres = 1.1;
  int y2diff = ch->y2() - neib->y2();
  int y1diff = ch->y1() - neib->y1();
  if (MatchY1(neib, ench)) {
    if (abs(y2diff) < min(ch->Height(), neib->Height()) * kDiffRatioThres) {
      if (y2diff < 0) {
        if (neib->style() == EnChar::STYLEa
            || neib->style() == EnChar::STYLEh) {
          ench->set_style(neib->style());
          ch_line->UpdateLayout(EnChar::BOTTOM_MARGIN);
          return true;
        }
        if (neib->style() == EnChar::STYLEy
            || neib->style() == EnChar::STYLEf) {
          ench->set_style(
              EnChar::ParseStyle(neib->style() & ~EnChar::BOTTOM_MARGIN));
          return true;
        }
      } else if (neib->style() == EnChar::STYLEa
          || neib->style() == EnChar::STYLEh) {
        ench->set_style(neib->style());
        ench->UpdateLayout(EnChar::BOTTOM_MARGIN);
        return true;
      }
    }
  } else if (MatchY2(neib, ench)) {
    if (abs(y1diff) < min(ch->Height(), neib->Height()) * kDiffRatioThres) {
      if (y1diff > 0) {
        if (neib->style() == EnChar::STYLEa
            || neib->style() == EnChar::STYLEy) {
          ench->set_style(neib->style());
          ch_line->UpdateLayout(EnChar::TOP_MARGIN);
          return true;
        }
        if (neib->style() == EnChar::STYLEh
            || neib->style() == EnChar::STYLEf) {
          ench->set_style(
              EnChar::ParseStyle(neib->style() & ~EnChar::TOP_MARGIN));
          return true;
        }
      } else if (neib->style() == EnChar::STYLEa
          || neib->style() == EnChar::STYLEy) {
        ench->set_style(neib->style());
        ench->UpdateLayout(EnChar::TOP_MARGIN);
        return true;
      }
    }
  } else {
    const int kDiffHThres = min(ch->Height(), ch_line->min_char_h())
        * ((cl->CharNum() == 1 && ch->x2() > cl->x2()) ? 0.8 : 1.1);
    if (abs(y1diff) < kDiffHThres && abs(y2diff) < kDiffHThres) {
      if (y1diff < 0 && y2diff > 0 && neib->style() == EnChar::STYLEa) {
        ench->set_style(EnChar::STYLEf);
        return true;
      }
      if (y1diff > 0 && y2diff > 0) {
        if (neib->style() == EnChar::STYLEh) {
          ench->set_style(EnChar::STYLEy);
          return true;
        }
        if (neib->style() == EnChar::STYLEa) {
          ench->set_style(EnChar::STYLEy);
          ch_line->UpdateLayout(EnChar::TOP_MARGIN);
          return true;
        }
      }
      if (y1diff < 0 && y2diff < 0) {
        if (neib->style() == EnChar::STYLEy) {
          ench->set_style(EnChar::STYLEh);
          return true;
        }
        if (neib->style() == EnChar::STYLEa) {
          ch_line->UpdateLayout(EnChar::BOTTOM_MARGIN);
          ench->set_style(EnChar::STYLEh);
          return true;
        }
      }
      if (y1diff > 0 && y2diff < 0) {
        if (neib->style() == EnChar::STYLEf) {
          ench->set_style(EnChar::STYLEa);
          return true;
        }
        if (neib->style() == EnChar::STYLEa) {
          ch_line->UpdateLayout(EnChar::TOP_MARGIN);
          ch_line->UpdateLayout(EnChar::BOTTOM_MARGIN);
          ench->set_style(EnChar::STYLEa);
          return true;
        }
      }
    }
  }
  return false;
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
      if (itl->CharNum() > jtl->CharNum()) {
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

CharLine* ConnCompBased::CreateAttrEnCharLine(CharConstItr begin,
                                              CharConstItr end) {
  CharLine* cl = new CharLine();
  for (; begin != end; ++begin) {
    cl->AddChar(new AttrEnChar(static_cast<EnChar*>(*begin)->GetCC()));
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
        tllist->insert(itr, CreateAttrEnCharLine(begin, bit));
        begin = bit;
      }
    }

    if (begin != clist.begin()) {
      if (begin != end) {
        tllist->insert(itr, CreateAttrEnCharLine(begin, end));
      }
      cout << "split " << (cl)->ToString() << endl;
      delete cl;
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
      delete *itr;
      itr = tllist->erase(itr);
    } else {
      ++itr;
    }
  }
}
