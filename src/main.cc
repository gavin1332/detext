#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common/test.h"
#include "common/core.h"
#include "common/utils.h"
#include "evaluate/evaluate.h"
#include "evaluate/icdar.h"
#include "evaluate/icdar2005.h"
#include "evaluate/icdar2011.h"
#include "td/liuyi13.h"
#include "td/frangi98.h"

using namespace std;
using namespace cv;

namespace dtxt {
extern bool PRINT_LOG_;
extern bool SHOW_GRAY_;
extern bool SHOW_RESPONSE_;
extern bool SHOW_GROUP_STEP_;
extern bool SHOW_GROUPED_RESULT_;
extern bool SHOW_FINAL_;

extern double THRESHOLD_;

bool PRINT_LOG_ = true;
bool SHOW_GRAY_ = true;
bool SHOW_RESPONSE_ = true;
bool SHOW_GROUP_STEP_ = true;
bool SHOW_GROUPED_RESULT_ = true;
bool SHOW_FINAL_ = true;

double THRESHOLD_ = 3;
}

using namespace dtxt;

void SaveEList(const list<TextLine*>& E_list, const String& img_path) {
  ofstream writer(img_path + ".txt");
  list<TextLine*>::const_iterator itr = E_list.begin();
  for (; itr != E_list.end(); ++itr) {
    writer << (*itr)->x1() << " " << (*itr)->y1() << " " << (*itr)->x2() << " "
        << (*itr)->y2() << endl;
  }
  writer.close();
  TestUtils::Print("Saved estimated rectangles\n");
}

void ReadRects(const String& img_path, vector<Rect>* rect_vec) {
  ifstream in(img_path + ".txt");
  if (in.fail())
    return;

  vector<string> piece_vec;
  string line;
  while (true) {
    getline(in, line);
    if (line.length() == 0)
      break;

    piece_vec.clear();
    CmnUtils::Split(line, ' ', &piece_vec);

    int x1 = atoi(piece_vec[0].c_str());
    int y1 = atoi(piece_vec[1].c_str());
    int x2 = atoi(piece_vec[2].c_str());
    int y2 = atoi(piece_vec[3].c_str());
    Rect rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

    rect_vec->push_back(rect);
  }

  in.close();
}

void ReleaseList(list<TextLine*>* tllist) {
  list<TextLine*>::iterator it = tllist->begin(), end = tllist->end();
  for (; it != end; ++it) {
    delete *it;
  }
  tllist->clear();
}

void PrintProgress(int count, int size) {
  cout << "Progress:" << (float) count * 100 / size << "%" << endl;
}

#if 1

int main(int argc, char** argv) {
  const string base_dir = "/home/liuyi/project/cpp/testdata/scene/2011";
  ICDAR* icdar = new ICDAR2011(base_dir);

  const string data_dir = icdar->test_data_dir();
  vector<string> filename_vec;
  CmnUtils::RetrieveFilenames(data_dir, ".jpg", &filename_vec, true);
  sort(filename_vec.begin(), filename_vec.end());

//  Frangi98 detector;
  LiuYi13 detector;
  Evaluator evaluator;
  vector<string>::iterator it = filename_vec.begin();
  int count = 1;
  const int file_count_total = filename_vec.size();
  double exec_time = (double) getTickCount();
  for (; it != filename_vec.end(); ++it, ++count) {
    if (it->compare("103.jpg") < 0) {
      continue;
    }
    THRESHOLD_ = 5;

    const string img_path = data_dir + "/" + *it;
    TestUtils::Print(img_path);
    PrintProgress(count, file_count_total);

    Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);

    vector<Rect> rect_vec;
    ReadRects(img_path, &rect_vec);
    bool notified = TestUtils::ShowRects(img, rect_vec, Scalar(255, 255, 255));
    if (notified) {
      continue;
    }

    float zoom = 1;
    if (img.cols > 1280) {
      zoom = (float) 1280 / img.cols;
      resize(img, img, Size(), zoom, zoom);
    }

//    Rect rect;
//    bool success = TestUtils::UserDrawRect(img, &rect);
//    if (!success) continue;
//    img = img(rect);

    const list<TextLine*>& T_list = icdar->RetrieveTList(img_path);
    list<TextLine*>* E_list = new list<TextLine*>;
    detector.Detect(img, E_list);
    if (zoom < 1) {
      list<TextLine*>* new_E_list = new list<TextLine*>;
      list<TextLine*>::iterator it = E_list->begin();
      for (; it != E_list->end(); ++it) {
        new_E_list->push_back((*it)->TuneRect(1 / zoom));
      }
      ReleaseList(E_list);
      delete E_list;
      E_list = new_E_list;
    }

    evaluator.RecordMatch(*E_list, T_list);

//    SaveEList(*E_list, img_path);
    ReleaseList(E_list);
    delete E_list;
  }

  exec_time = ((double) getTickCount() - exec_time) / getTickFrequency();
  TestUtils::Print("Time cost/s", exec_time);

  float precision;
  float recall;
  float f_measure;
  evaluator.Report(&precision, &recall, &f_measure);

  TestUtils::Print("precision", precision);
  TestUtils::Print("recall", recall);
  TestUtils::Print("f_measure", f_measure);

  delete icdar;

  return EXIT_SUCCESS;
}

#elif !1

int main() {
  const string base_dir = "/home/liuyi/project/cpp/testdata/scene/2011";
  const string img_path = base_dir + "/test-textloc-gt/test-textloc-gt/123.jpg";
  Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);
  Frangi98 detector;
  list<TextLine*> result;
  detector.Detect(img, &result);
  return 0;
}

#else

int main() {
  Mat a = Mat::ones(3, 3, CV_64FC1);
  Mat mask = Mat::ones(3, 3, CV_8UC1);
  mask.at<uchar>(1, 1) = 0;
  cout << a << endl;
  Mat b;
  a.copyTo(a, mask);
  cout << a << endl;
  return 0;
}

#endif
