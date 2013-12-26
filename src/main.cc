#include <string>

#include "evaluate/icdar2005.h"
#include "evaluate/icdar2011.h"
#include "integrate/instant.h"
#include "integrate/precomputed.h"
#include "td/liuyi13.h"

using namespace std;
using namespace cv;

using namespace dtxt;

int main() {
  const string base_dir = "/home/liuyi/project/cpp/testdata/scene/2011";
  ICDAR2011 dataset(base_dir);
  LiuYi13 detector;
  InstantProcess process(&dataset, &detector);
//  PreComputedProcess process(&dataset);
  process.Run();

  return EXIT_SUCCESS;
}
