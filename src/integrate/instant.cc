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

void InstantProcess::Run() {
  extern bool SHOW_PRECOMPUTED_RESULT_;
  extern bool SAVE_RESULT_INTERACTION_;

  const string& data_dir = dataset_->test_data_dir();

  vector<string> filename_vec;
  dataset_->RetrieveImgIds(&filename_vec);

  const int total_file = filename_vec.size();
  int count = 0;
  double exe_time = (double) getTickCount();
  for (const string& filename : filename_vec) {
    const string img_path = data_dir + "/" + filename;
    TestUtils::Print(img_path);
    PrintProgress(++count, total_file);

    // 145, 184, 186, 187, 188, 364, 371 and 398 is very hard
    if (filename.compare("101.jpg") < 0) {
      continue;
    }

    Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);
    if (SHOW_PRECOMPUTED_RESULT_ && ShowPreComputedResult(img, filename)) {
      continue;
    }

    double zoom = 1;
    if (img.rows > 1280 || img.cols > 1280) {
      zoom = (double) 1280 / img.cols;
      resize(img, img, Size(), zoom, zoom);
    }

    list<TextLine*>* estlist = new list<TextLine*>;
    detector_->Detect(img, estlist);
    if (zoom < 1) {
      ZoomTextLine(&estlist, 1 / zoom);
    }
    dataset_->IncEvaluate(filename, *estlist);

    if (SAVE_RESULT_INTERACTION_
        && IntegrateUtils::ShowTextLines("Save result? [y/n]", img, *estlist)) {
      dataset_->SaveEstResult(filename, *estlist);
      TestUtils::Print("Estimated results saved");
    }

    IntegrateUtils::ReleaseList(estlist);
    delete estlist;
  }

  Report(exe_time);
}

void InstantProcess::ZoomTextLine(list<TextLine*>** estlist, double factor) {
  list<TextLine*>* new_estlist = new list<TextLine*>;
  for (TextLine* tl : **estlist) {
    new_estlist->push_back(tl->NewZoomedTextLine(factor));
  }
  IntegrateUtils::ReleaseList(*estlist);
  delete *estlist;
  *estlist = new_estlist;
}

bool InstantProcess::ShowPreComputedResult(const Mat& img,
                                           const string& filename) {
  list<TextLine*> tllist;
  dataset_->RetrieveTextLines(filename, &tllist);
  bool jump = IntegrateUtils::ShowTextLines("Jump to the next? [y/n]", img,
                                            tllist);
  IntegrateUtils::ReleaseList(&tllist);
  return jump;
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
