#ifndef INTEGRATE_OFFLINE_H_
#define INTEGRATE_OFFLINE_H_

#include "evaluate/dataset.h"

namespace dtxt {

class OfflineProcess {
 public:
  OfflineProcess(DataSet* dataset)
      : dataset_(dataset) {
  }

  virtual ~OfflineProcess() {
  }

  virtual void Run() = 0;

 protected:
  DataSet* dataset_;

};

}

#endif
