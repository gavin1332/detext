#ifndef INTEGRATE_PRECOMPUTED_H_
#define INTEGRATE_PRECOMPUTED_H_

#include "offline.h"
#include "evaluate/dataset.h"

namespace dtxt {

class PreComputedProcess : public OfflineProcess {
 public:
  PreComputedProcess(DataSet* dataset)
      : OfflineProcess(dataset) {
  }

  void Run();

};

}

#endif
