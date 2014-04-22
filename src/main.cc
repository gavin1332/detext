#include <string>

#include "evaluate/icdar2005.h"
#include "evaluate/icdar2011.h"
#include "integrate/instant.h"
#include "integrate/precomputed.h"
#include "td/liuyi14.h"

using namespace std;
using namespace cv;

using namespace dtxt;

namespace dtxt {

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

bool PRINT_LOG_ = true;
bool SHOW_GRAY_ = true;
bool SHOW_PRECOMPUTED_RESULT_ = true;
bool SHOW_RESPONSE_ = true;
bool SHOW_GROUP_STEP_ = !true;
bool SHOW_GROUPED_RESULT_ = !true;
bool SHOW_FINAL_ = !true;
bool SPLIT_CHAR_LINE_ = !true;
bool SAVE_RESULT_INTERACTION_ = !true;

double THRESHOLD_ = 3;

}

#if 1

int main() {
  const string base_dir = "/home/liuyi/project/cpp/testdata/scene/2011";
  ICDAR2011 dataset(base_dir);
  LiuYi14 detector;
  InstantProcess process(&dataset, &detector);
//  PreComputedProcess process(&dataset);
  process.Run();

  return EXIT_SUCCESS;
}

#else

int main() {
  const string base_dir = "/home/liuyi/project/cpp/testdata/scene/2011";
  ICDAR2011 dataset(base_dir);
  LiuYi13 detector;
  InstantProcess process(&dataset, &detector);
//  PreComputedProcess process(&dataset);
  process.Run();

  return EXIT_SUCCESS;
}

#endif
