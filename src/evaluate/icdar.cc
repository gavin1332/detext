#include "evaluate/icdar.h"

using namespace std;
using namespace dtxt;

ICDAR::~ICDAR() {
  map<string, list<TextLine*>*>::iterator mit = T_list_map_.begin();
  for (; mit != T_list_map_.end(); ++mit) {
    list<TextLine*>::iterator lit = mit->second->begin(), lend = mit->second
        ->end();
    for (; lit != lend; ++lit) {
      delete *lit;
    }
    delete mit->second;
  }
}
