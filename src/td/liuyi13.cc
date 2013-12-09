#include "td/liuyi13.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common/test.h"

using namespace std;
using namespace cv;
using namespace dtxt;

void LiuYi13::Detect(const cv::Mat& img, std::list<TextLine*>* tllist) {
  extern bool SHOW_GRAY_;
  extern bool SHOW_FINAL_;

  Mat gray;
  cvtColor(img, gray, CV_BGR2GRAY);
  if (SHOW_GRAY_) {
    TestUtils::ShowImage(gray);
  }

  Mat resp[2];
  genRespMap(gray, resp);

  list<ConnComp*> cclist[2];
  ParseResp(gray, resp, cclist);

  for (int i = 0; i < 2; ++i) {
    CheckCCValidation(cclist + i);
    GroupConnComp(gray, cclist + i, tllist);
    ReleaseCCList(cclist + i);
  }

  if (SHOW_FINAL_) {
    DispRects(img, *tllist, Scalar(255, 255, 255));
  }
}

void LiuYi13::ReleaseCCList(list<ConnComp*>* cclist) {
  list<ConnComp*>::iterator cc = cclist->begin();
  for (; cc != cclist->end(); ++cc) {
    delete *cc;
  }
  cclist->clear();
}

void LiuYi13::ParseResp(const Mat& gray, const Mat* resp,
                        list<ConnComp*>* cclist) {
  extern bool SHOW_RESPONSE_;
  // TODO
  extern double THRESHOLD_;

  // TODO: dynamically choose
  const double kBinThres = THRESHOLD_;
  Mat mask[2];
  mask[POS] = resp[POS] < -kBinThres;
  mask[NEG] = resp[NEG] > kBinThres;
  if (SHOW_RESPONSE_) {
    TestUtils::ShowImage(mask[POS]);
    TestUtils::ShowImage(mask[NEG]);
  }

  const char kTmpFill = 1;
  for (int i = 0; i < 2; ++i) {
    MatIterator_<uchar> itr = mask[i].begin<uchar>();
    MatIterator_<uchar> end = mask[i].end<uchar>();
    for (; itr != end; ++itr) {
      if (*itr != 255)
        continue;

      Rect rect;
      floodFill(mask[i], itr.pos(), kTmpFill, &rect, 0, 0, 8);
      ConnComp* cc = new AttrConnComp();
      for (int y = 0; y < rect.height; ++y) {
        uchar* mask_ptr = mask[i].ptr<uchar>(rect.y + y, rect.x);
        const uchar* gray_ptr = gray.ptr<uchar>(rect.y + y, rect.x);
        const double* resp_ptr = resp[i].ptr<double>(rect.y + y, rect.x);
        for (int x = 0; x < rect.width; ++x) {
          if (mask_ptr[x] == kTmpFill) {
            cc->AddPixel(
                new AttrPix(Point(rect.x + x, rect.y + y), gray_ptr[x],
                            resp_ptr[x]));
          }
        }
      }
      cclist[i].push_back(cc);
      floodFill(mask[i], itr.pos(), 0, &rect, 0, 0, 8);
    }
  }
}

void LiuYi13::CheckCCValidation(list<ConnComp*>* cclist) {
  list<ConnComp*>::iterator itr = cclist->begin();
  for (; itr != cclist->end(); ++itr) {
    (*itr)->CheckValidation();
  }
}

void LiuYi13::genRespMap(const Mat& gray, Mat* resp) {
  Mat fgray;
  gray.convertTo(fgray, CV_64FC1);

  const double kInitW = 3;
  const double kSigmaK = sqrt(2);
  double sigma = kInitW / 2 / sqrt(3.0);
  double delta_sigma = sigma;
  const int kN = 12;
  Mat gauss = fgray.clone();
  Mat last_gauss, dog;
  Mat accum_map[2], mask[2];
  for (int i = 0; i < 2; ++i) {
    mask[i] = Mat::ones(gauss.size(), CV_8UC1) * 255;
  }

  const double kMaskThres = 0.8;
  for (int i = 0; i < kN; ++i) {
    cout << "i: " << i << endl;
    last_gauss = gauss.clone();
    GaussianBlur(gauss, gauss, Size(0, 0), delta_sigma);
    if (i >= 1) {
      dog = gauss - last_gauss;
      if (i == 1) {
        accum_map[POS] = dog.clone();
        accum_map[NEG] = dog.clone();
      } else {
        accum_map[POS] = min(dog, accum_map[POS]);
        accum_map[NEG] = max(dog, accum_map[NEG]);
      }
      accum_map[POS].copyTo(resp[POS], mask[POS]);
      accum_map[NEG].copyTo(resp[NEG], mask[NEG]);
      mask[POS] &= 255 - (accum_map[NEG] > kMaskThres);
      mask[NEG] &= 255 - (accum_map[POS] < -kMaskThres);
      double tmax, tmin;
      Point pmax, pmin;
      minMaxLoc(dog, &tmin, &tmax, &pmin, &pmax);
      cout << "max resp: " << tmin << " " << tmax << endl;
      cout << "pos: " << pmin << " " << pmax << endl;
    }
    delta_sigma = sigma * (kSigmaK - 1);
    sigma *= kSigmaK;
  }
}
