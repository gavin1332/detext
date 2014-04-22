#include "td/liuyi14.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common/utils.h"
#include "common/test.h"

using namespace std;
using namespace cv;
using namespace dtxt;

void LiuYi14::Detect(const Mat& img, list<TextLine*>* tllist) {
  extern bool SHOW_GRAY_;
  extern bool SHOW_RESPONSE_;
//  extern bool SHOW_GROUPED_RESULT_;
//  extern bool SHOW_FINAL_;
//  extern bool SPLIT_CHAR_LINE_;

  Mat gray;
  cvtColor(img, gray, CV_BGR2GRAY);
//  Mat channel[3];
//  cv::split(img, channel);
//  gray = channel[0];
  if (SHOW_GRAY_) {
    TestUtils::ShowImage(gray);
  }

  Mat resp[2];
  GenRespMap(gray, resp);
  if (SHOW_RESPONSE_) {
    Mat mat;
    normalize(-resp[POS], mat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    TestUtils::ShowImage(mat);
    normalize(resp[NEG], mat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    TestUtils::ShowImage(mat);

    TestUtils::ShowImage((resp[POS] < -0.8) & (resp[NEG] > 0.8));
  }
}

void haha(const Mat& dog, Mat* accum, Mat* mask, Polarity pol) {
  const int width = dog.cols;
  const int height = dog.rows;
  for (int y = 0; y < height; ++y) {
    const double* dog_ptr = dog.ptr<double>(y);
    double* accum_ptr = accum[pol].ptr<double>(y);
    uchar* mask_ptr = mask[pol].ptr<uchar>(y);
    for (int x = 0; x < width; ++x) {
      if (mask_ptr[x] == 0
          || (pol == POS && MathUtils::CmpInAbsErr(dog_ptr[x], 0) >= 0)
          || (pol == NEG && MathUtils::CmpInAbsErr(dog_ptr[x], 0) <= 0)) {
        continue;
      }

      double cmp = fabs(dog_ptr[x]) - fabs(accum_ptr[x]);
      if (cmp > 0.001) {
        accum_ptr[x] = dog_ptr[x];
      } else if (cmp < 0.001) {
        if (mask_ptr[x] != 1) {
          mask_ptr[x] = 1;
        } else {
          mask_ptr[x] = 0;
        }
      }
    }
  }
}

void LiuYi14::GenRespMap(const Mat& gray, Mat* resp) {
  Mat gauss;
  gray.convertTo(gauss, CV_64FC1);
  GaussianBlur(gauss, gauss, Size(0, 0), 0.5);  // TODO

  const double kSigmaK = sqrt(2);
  const int kN = 11;
  Mat accum_map[2], mask[2];
  for (int i = 0; i < 2; ++i) {
    mask[i] = Mat::ones(gauss.size(), CV_8UC1) * 255;
  }
  Mat last_gauss, dog;
  double sigma = 1.4;
  double delta_sigma = sigma * (kSigmaK - 1);
  for (int i = 1; i < kN; ++i) {
    TestUtils::Log("index", i);
    last_gauss = gauss.clone();
    GaussianBlur(gauss, gauss, Size(0, 0), delta_sigma);
    dog = gauss - last_gauss;
    if (i == 1) {
      dog.copyTo(accum_map[POS], dog < 0);
      dog.copyTo(accum_map[NEG], dog > 0);
    } else {
      haha(dog, accum_map, mask, POS);
      haha(dog, accum_map, mask, NEG);
    }
    delta_sigma = sigma * (kSigmaK - 1);
    sigma *= kSigmaK;
  }

  resp[POS] = accum_map[POS];
  resp[NEG] = accum_map[NEG];
}
