#include "common/test.h"

#ifdef INNER_TEST

#include <stdarg.h>
#include <ctime>
#include <cctype>

#include <iostream>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dtxt;

bool TestUtils::ShowImage(const char* win_name, const Mat& img, int wait_time) {
  namedWindow(win_name);

  setMouseCallback(win_name, CmnMouseCallback);

  imshow(win_name, img);

  bool notified = UserInteraction(win_name, img, wait_time);

//  destroyWindow(win_name);

  return notified;
}

inline void TestUtils::CmnMouseCallback(int event, int x, int y, int flags,
                                        void *param) {
  switch (event) {
    case CV_EVENT_LBUTTONUP:
      std::cout << "(x, y): (" << x << ", " << y << ")" << std::endl;
      break;
  }
}

void TestUtils::CropMouseCallback(int event, int x, int y, int flags,
                                  void *param) {
  CropParam* crop_param = static_cast<CropParam*>(param);
  switch (event) {
    case CV_EVENT_LBUTTONDOWN:
      crop_param->end = crop_param->start = Point(x, y);
      crop_param->drawing = true;
      break;
    case CV_EVENT_MOUSEMOVE:
      if (crop_param->drawing) {
        crop_param->end = Point(x, y);
      }
      break;
    case CV_EVENT_LBUTTONUP:
      if (crop_param->drawing) {
        crop_param->drawing = false;
      }
      break;
    case CV_EVENT_RBUTTONDOWN:
      crop_param->finished = true;
      break;
  }
}

bool TestUtils::UserDrawRect(const Mat& gray_rgb, Rect* rect) {
  static const char kEsc = 27;
  static const string kWinName = "Draw a rectangle";

  Mat rgb;
  if (gray_rgb.channels() == 3) {
    rgb = gray_rgb;
  } else {
    cvtColor(gray_rgb, rgb, CV_8UC3);
  }

  namedWindow(kWinName);

  CropParam param;
  param.start = cv::Point(-1, -1);
  param.end = cv::Point(-1, -1);
  param.drawing = false;
  param.finished = false;
  setMouseCallback(kWinName, CropMouseCallback, &param);

  Mat temp;
  int signal = 0;
  while (!param.finished && signal != kEsc) {
    temp = rgb.clone();
    rectangle(temp, param.start, param.end, Scalar(255, 255, 255));
    imshow(kWinName, temp);
    signal = waitKey(100);
  }

  if (param.finished) {
    rect->x = min(param.start.x, param.end.x);
    rect->y = min(param.start.y, param.end.y);
    rect->width = max(param.start.x, param.end.x) - rect->x + 1;
    rect->height = max(param.start.y, param.end.y) - rect->y + 1;
  }

  destroyWindow(kWinName);

  return signal != kEsc;
}

bool TestUtils::UserInteraction(const char* win_name, const Mat& img,
                                int wait_time) {
  static const char WIN_CLOSED = static_cast<char>(-1);
  static const char BLANK = ' ';

#ifdef __MINGW32__
  static const char ENTER = 13;
  static const char NUM_ENTER = 13;
#else
  static const char ENTER = '\n';
  static const char NUM_ENTER = -115;
#endif

  static const char PLUS = 61;
  static const char SUBS = 45;
  static const char RESUME = '0';

  static const char ESC = 27;
  static const char SAVE = 's';
  static const char NOTIFY = 'n';

  char signal = '\0';
  float ratio = 1;
  Mat curr = img;
  while (signal != BLANK && signal != ENTER && signal != NUM_ENTER
      && signal != WIN_CLOSED) {
    signal = waitKey(wait_time);

    switch (signal) {
      case RESUME:
      case PLUS:
      case SUBS: {
        if ((ratio < 0.3f && signal == SUBS)
            || (ratio > 3.9f && signal == PLUS)) {
          break;
        }

        if (signal == RESUME)
          ratio = 1;
        if (signal == PLUS)
          ratio += 0.5;
        if (signal == SUBS)
          ratio -= 0.2f;

        int method = (ratio < 1) ? INTER_LINEAR : INTER_NEAREST;
        resize(img, curr, Size(0, 0), ratio, ratio, method);

        imshow(win_name, curr);

        break;
      }

      case SAVE: {
        time_t file_prefix = time(NULL) % 1000000;
        const char* file_postfix = ".png";

        stringstream ss;
        ss << file_prefix << file_postfix;
        imwrite(ss.str(), curr);
        break;
      }

      case NOTIFY:
        return true;

      case ESC: {
        Print("user quits");
        destroyWindow(win_name);
        exit(0);
      }
    }
  }

  return false;
}

Mat TestUtils::PackManyImages(const Mat* imgs, const int N) {
  // If the number of arguments is lesser than 0 or greater than 12
  // return without showing
  if (N <= 0 || N > MAX_IMAGE_NUM) {
    cout << "The number of images is invalid, so ignores" << endl;
    return Mat();
  }

  // The size of embedded images
  int inner_size;
  // The row and column numbers of embedded images
  int rows, cols;
  CalcLayout(N, &rows, &cols, &inner_size);

  const int inner_space = 10;
  Mat canvas = CreateCanvas(rows, cols, inner_size, inner_space);

  int index = 0;
  Point pos(inner_space, inner_space);
  while (index < N) {
    Resize2Canvas(imgs[index], &canvas, pos, inner_size);

    ++index;

    // Update the next position
    if (index % cols == 0) {
      pos.x = inner_space;
      pos.y += inner_space + inner_size;
    } else {
      pos.x += inner_space + inner_size;
    }
  }

  return canvas;
}

void TestUtils::ShowGradient(const Mat* grad, float mag_shield, bool white_bg) {
  Mat rgb, hsv;

  Mat chnl[3];
  grad[1].convertTo(chnl[0], CV_8UC1, 180);
  Mat mask = grad[0] >= mag_shield;

  chnl[1] = mask;

  double min, max;
  minMaxLoc(grad[0], &min, &max);
  grad[0].convertTo(chnl[2], CV_8UC1, 255 / max);
  if (white_bg)
    chnl[2] += Scalar(255) - mask;

  merge(chnl, 3, hsv);
  cvtColor(hsv, rgb, CV_HSV2BGR);

  ShowImage(rgb);
}

void TestUtils::ShowHistogram(const int* data, int len, int zoom) {
  float fdata[len];

  long accum = 0;
  for (int i = 0; i < len; ++i) {
    accum += data[i];
  }

  for (int i = 0; i < len; ++i) {
    fdata[i] = (float) data[i] / accum;
  }

  ShowNormedHistogram(fdata, len, zoom);
}

void TestUtils::ShowNormedHistogram(const float* data, int len, int zoom) {
  if (zoom > 1) {
    int size = len * zoom;
    float *new_data = new float[size];

    for (int i = 0; i < size; ++i) {
      new_data[i] = data[i / zoom] / zoom;
    }

    ShowNormedHistogram(new_data, size, 1);

    delete[] new_data;
  } else if (zoom < -1) {
    int scale = 0 - zoom;
    int size = (len + scale - 1) / scale;
    float *new_data = new float[size];
    memset(new_data, 0, size * sizeof(float));

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < scale && scale * i + j < len; j++) {
        new_data[i] += data[scale * i + j];
      }
    }

    ShowNormedHistogram(new_data, size, 1);

    delete[] new_data;
  } else {
    int height = len * 2 / 3;
    if (height == 0)
      height = 1;

    Mat img(height, len, CV_8UC1);

    float highest = 0;
    float *origin_height = new float[len];
    for (int i = 0; i < len; i++) {
      origin_height[i] = data[i] * height;
      highest = (highest > origin_height[i]) ? highest : origin_height[i];
    }

    // the factor of the highest origin height to reach 4/5 of the image height
    float factor = height * 4 / (5 * highest);

    for (int i = 0; i < len; i++) {
      int target_height = cvRound(origin_height[i] * factor);

      int j = 0;
      while (j < target_height) {
        img.at<uchar>(height - j - 1, i) = 0;
        ++j;
      }
      while (j < height) {
        img.at<uchar>(height - j - 1, i) = 255;
        ++j;
      }
    }

    ShowImage(img);

    delete[] origin_height;
  }
}

bool TestUtils::ShowRect(const Mat& gray_rgb, const Rect& rect, Scalar bgr,
                         int thickness) {
  Mat color;
  if (gray_rgb.channels() == 1) {
    cvtColor(gray_rgb, color, CV_GRAY2BGR);
  } else {
    assert(gray_rgb.channels() == 3);
    color = gray_rgb.clone();
  }

  rectangle(color, rect, bgr, thickness);
  return ShowImage(color);
}

bool TestUtils::ShowRects(const Mat& gray_rgb, const vector<Rect>& rect_vec,
                          Scalar bgr) {
  Mat color;
  if (gray_rgb.channels() == 1) {
    cvtColor(gray_rgb, color, CV_GRAY2BGR);
  } else {
    color = gray_rgb.clone();
  }

  vector<Rect>::const_iterator it = rect_vec.begin();
  for (; it != rect_vec.end(); ++it) {
    rectangle(color, *it, bgr, 2);
  }
  return ShowImage(color);
}

bool TestUtils::ShowManyImages(const char* win_name, const Mat* imgs,
                               const int N, int wait_time) {
  Mat canvas = PackManyImages(imgs, N);

  bool notified = false;
  if (!canvas.empty()) {
    notified = ShowImage(win_name, canvas, wait_time);
  }

  return notified;
}

bool TestUtils::ShowManyImages(const char *win_name, int argc, ...) {
  // Used to get the arguments passed
  va_list args;
  va_start(args, argc);

  Mat mats[argc];
  int index = 0;
  while (index < argc) {
    // Get the pointer of the image
    const Mat* img = va_arg(args, const Mat*);
    mats[index] = *img;

    ++index;
  }

  bool notified = ShowManyImages(win_name, mats, argc, 0);

  // End the number of arguments
  va_end(args);

  return notified;
}

void TestUtils::CalcLayout(int argc, int* rows, int* cols, int* inner_size) {
  if (argc == 1) {
    *rows = *cols = 1;
    *inner_size = 400;
  } else if (argc == 2) {
    *rows = 1;
    *cols = 2;
    *inner_size = 400;
  } else if (argc == 3 || argc == 4) {
    *rows = 2;
    *cols = 2;
    *inner_size = 400;
  } else if (argc == 5 || argc == 6) {
    *rows = 2;
    *cols = 3;
    *inner_size = 200;
  } else if (argc == 7 || argc == 8) {
    *rows = 2;
    *cols = 4;
    *inner_size = 150;
  } else {
    *rows = 3;
    *cols = 4;
    *inner_size = 150;
  }
}

Mat TestUtils::CreateCanvas(int rows, int cols, int inner_size,
                            int inner_space) {
  Size win_size((inner_space + inner_size) * cols + inner_space,
                (inner_space + inner_size) * rows + inner_space);
  Mat canvas(win_size, CV_8UC3, Scalar(100, 100, 100));

  return canvas;
}

void TestUtils::Resize2Canvas(const Mat& img, Mat* canvas, Point pos,
                              int inner_size) {
  // Find the scaling factor to resize the image
  int max = (img.cols > img.rows) ? img.cols : img.rows;
  float scale = max * 1.0f / inner_size;

  // Get the ROI on the canvas
  Rect rect(pos.x, pos.y, static_cast<int>(img.cols / scale),
            static_cast<int>(img.rows / scale));
  Mat roi(*canvas, rect);

  // Resize the input image to the canvas
  if (img.channels() == 1) {  // Single channel image
    Mat rgb(roi.size(), CV_8UC3);
    cvtColor(img, rgb, CV_GRAY2RGB);
    resize(rgb, roi, roi.size());
  } else {
    assert(img.channels() == 3);
    resize(img, roi, roi.size());
  }
}

bool TestUtils::ShowLine(const Mat& gray_rgb, Point p1, Point p2, Scalar bgr) {
  Mat color;
  if (gray_rgb.channels() == 1) {
    cvtColor(gray_rgb, color, CV_GRAY2BGR);
  } else {
    assert(gray_rgb.channels() == 3);
    color = gray_rgb;
  }

  line(color, p1, p2, bgr);
  return ShowImage(color);
}

bool TestUtils::ShowPoint(const Mat& gray_rgb, Point point, Scalar bgr) {
  Mat color;
  if (gray_rgb.channels() == 1) {
    cvtColor(gray_rgb, color, CV_GRAY2BGR);
  } else {
    assert(gray_rgb.channels() == 3);
    color = gray_rgb;
  }

  color.at<Vec3b>(point) = Vec3b(bgr[0], bgr[1], bgr[2]);
  return ShowImage(color);
}

bool TestUtils::ShowPoints(const Mat& gray_rgb, vector<Point> point_vec,
                           Scalar bgr) {
  Mat color;
  if (gray_rgb.channels() == 1) {
    cvtColor(gray_rgb, color, CV_GRAY2BGR);
  } else {
    assert(gray_rgb.channels() == 3);
    color = gray_rgb;
  }

  vector<Point>::iterator itr = point_vec.begin();
  Vec3b value(bgr[0], bgr[1], bgr[2]);
  for (; itr != point_vec.end(); ++itr) {
    color.at<Vec3b>(*itr) = value;
  }
  return ShowImage(color);
}

#endif
