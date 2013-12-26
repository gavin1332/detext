#include "integrate/precomputed.h"

#include <string>
#include <list>
#include <fstream>

#include "common/test.h"
#include "common/core.h"
#include "common/utils.h"
#include "integrate/utils.h"

using namespace std;
using namespace cv;
using namespace dtxt;

void PreComputedProcess::Run() {
  const string data_dir = dataset_->test_data_dir();
  vector<string> filename_vec;
  CmnUtils::RetrieveFilenames(data_dir, ".jpg", &filename_vec, true);
  for (const string& filename : filename_vec) {
    list<TextLine*> estlist;
    dataset_->RetrieveTextLine(filename, &estlist);
    dataset_->IncEvaluate(filename, estlist);
    IntegrateUtils::ReleaseList(&estlist);
  }

  double precision, recall, f_measure;
  dataset_->Report(&precision, &recall, &f_measure);

  TestUtils::Print("precision", precision);
  TestUtils::Print("recall", recall);
  TestUtils::Print("f_measure", f_measure);
}
