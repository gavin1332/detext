#include "integrate/utils.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common/test.h"

using namespace std;
using namespace cv;
using namespace dtxt;

void IntegrateUtils::ReleaseList(list<TextLine*>* tllist) {
  for (TextLine* tl : *tllist) {
    delete tl;
  }
  tllist->clear();
}

bool IntegrateUtils::ShowTextLines(const string& title, const Mat& gray_rgb,
                                   const list<TextLine*>& tllist, Scalar bgr,
                                   int thickness) {
  Mat color;
  if (gray_rgb.channels() == 1) {
    cvtColor(gray_rgb, color, CV_GRAY2BGR);
  } else {
    color = gray_rgb.clone();
  }
  for (TextLine* tl : tllist) {
    rectangle(color, tl->ToCvRect(), bgr, thickness);
  }

  namedWindow(title);
  imshow(title, color);
  static const char ESC = 27;
  static const char WIN_CLOSED = static_cast<char>(-1);
  char signal = 0;
  while (signal != 'Y' && signal != 'y' && signal != 'N' && signal != 'n'
      && signal != ESC && signal != WIN_CLOSED) {
    signal = waitKey();
  }
  destroyWindow(title);

  if (signal == ESC || signal == WIN_CLOSED) {
    TestUtils::Print("user quits");
    exit(0);
  }
  return signal == 'Y' || signal == 'y';
}
