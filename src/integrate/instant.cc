#include "integrate/instant.h"

#include <string>
#include <vector>
#include <list>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "integrate/utils.h"
#include "common/test.h"
#include "common/core.h"
#include "common/utils.h"
#include "td/detext.h"
#include "evaluate/dataset.h"

using namespace std;
using namespace cv;
using namespace dtxt;

namespace dtxt {

extern bool EVALUATE_PRECOMPUTED_RESULT_;
extern bool PRINT_LOG_;
extern bool SHOW_PRECOMPUTED_RESULT_;
extern bool SHOW_GRAY_;
extern bool SHOW_RESPONSE_;
extern bool SHOW_GROUP_STEP_;
extern bool SHOW_GROUPED_RESULT_;
extern bool SHOW_FINAL_;
extern bool SAVE_RESULT_INTERACTION_;
extern bool SPLIT_CHAR_LINE_;

extern double THRESHOLD_;

bool EVALUATE_PRECOMPUTED_RESULT_ = true;
bool PRINT_LOG_ = true;
bool SHOW_GRAY_ = true;
bool SHOW_PRECOMPUTED_RESULT_ = true;
bool SHOW_RESPONSE_ = !true;
bool SHOW_GROUP_STEP_ = true;
bool SHOW_GROUPED_RESULT_ = true;
bool SHOW_FINAL_ = true;
bool SPLIT_CHAR_LINE_ = true;
bool SAVE_RESULT_INTERACTION_ = true;

double THRESHOLD_;

}

void InstantProcess::Run() {
  const string& data_dir = dataset_->test_data_dir();

  vector<string> filename_vec;
  dataset_->RetrieveImgIds(&filename_vec);

  const int total_file = filename_vec.size();
  int count = 0;
  double exe_time = (double) getTickCount();
  for (const string& filename : filename_vec) {
    const string img_path = data_dir + "/" + filename;
    TestUtils::Print(img_path);
    ++count;
    PrintProgress(count, total_file);

    // 145, 184, 186, 187, 188, 364, 371 and 398 is very hard
    if (filename.compare("145.jpg") < 0) {
      continue;
    }
    THRESHOLD_ = 5;
    SHOW_RESPONSE_ = !true;
    SHOW_GROUP_STEP_ = true;
    SPLIT_CHAR_LINE_ = true;
    SAVE_RESULT_INTERACTION_ = true;

    Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);
    if (SHOW_PRECOMPUTED_RESULT_) {
      list<TextLine*> tllist;
      dataset_->RetrieveTextLines(filename, &tllist);
      bool jump = IntegrateUtils::ShowTextLines("Jump to the next? [y/n]", img,
                                                tllist);
      IntegrateUtils::ReleaseList(&tllist);
      if (jump) {
        continue;
      }
    }

    double zoom = 1;
    if (img.rows > 1280 || img.cols > 1280) {
      zoom = (double) 1280 / img.cols;
      resize(img, img, Size(), zoom, zoom);
    }

    list<TextLine*>* estlist = new list<TextLine*>;
    detector_->Detect(img, estlist);
    if (zoom < 1) {
      list<TextLine*>* new_estlist = new list<TextLine*>;
      for (TextLine* tl : *estlist) {
        new_estlist->push_back(tl->NewZoomedTextLine(1 / zoom));
      }
      IntegrateUtils::ReleaseList(estlist);
      delete estlist;
      estlist = new_estlist;
    }
    dataset_->IncEvaluate(filename, *estlist);

    if (SAVE_RESULT_INTERACTION_) {
      if (IntegrateUtils::ShowTextLines("Save result? [y/n]", img, *estlist)) {
        dataset_->SaveEstResult(filename, *estlist);
        TestUtils::Print("Saved estimated rectangles");
      } else {
        TestUtils::Print("Result is not saved");
      }
    }

    IntegrateUtils::ReleaseList(estlist);
    delete estlist;
  }

  Report(exe_time);
}

void InstantProcess::Report(double exec_time) {
  exec_time = ((double) getTickCount() - exec_time) / getTickFrequency();
  TestUtils::Print("Time cost/s", exec_time);

  double precision, recall, f_measure;
  dataset_->Report(&precision, &recall, &f_measure);

  TestUtils::Print("precision", precision);
  TestUtils::Print("recall", recall);
  TestUtils::Print("f_measure", f_measure);

}

void InstantProcess::PrintProgress(int curr, int total) {
  cout << "Progress:" << (float) curr * 100 / total << "%" << endl;
}
