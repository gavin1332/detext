#include "td/detext.h"

#include "common/core.h"
#include "common/test.h"

using namespace std;
using namespace cv;
using namespace dtxt;

void TextDetector::DispRects(const Mat& gray, const list<TextLine*>& trlist,
                             Scalar color) {
  vector<Rect> rect_vec;
  list<TextLine*>::const_iterator itr = trlist.begin(), end = trlist.end();
  for (; itr != end; ++itr) {
    rect_vec.push_back((*itr)->ToCvRect());
  }
  TestUtils::ShowRects(gray, rect_vec, color);
}
