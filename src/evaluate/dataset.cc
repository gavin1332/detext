#include "evaluate/dataset.h"

#include <fstream>

#include "common/utils.h"

using namespace std;
using namespace dtxt;

const string DataSet::RESULT_POSTFIX = ".txt";

void DataSet::IncEvaluate(const std::string& id,
                          const std::list<dtxt::TextLine*>& estlist) {
  std::list<dtxt::TextLine*>* tgtlist = RetrieveTgtList(id);
  evaluator_.IncRecord(estlist, *tgtlist);
  PostProcess(&tgtlist);
}

void DataSet::SaveEstResult(const std::string& id,
                            const std::list<dtxt::TextLine*>& estlist) {
  ofstream writer(est_data_dir() + "/" + id + RESULT_POSTFIX);
  for (TextLine* tl : estlist) {
    writer << tl->x1() << " " << tl->y1() << " " << tl->x2() << " " << tl->y2()
           << endl;
  }
  writer.close();
}

void DataSet::RetrieveTextLines(const string& id, list<TextLine*>* tllist) {
  ifstream in(est_data_dir() + "/" + id + RESULT_POSTFIX);
  if (in.fail()) {
    return;
  }

  vector<string> piece_vec;
  string line;
  while (true) {
    getline(in, line);
    if (line.length() == 0) {
      break;
    }

    piece_vec.clear();
    CmnUtils::Split(line, ' ', &piece_vec);

    int x1 = atoi(piece_vec[0].c_str());
    int y1 = atoi(piece_vec[1].c_str());
    int x2 = atoi(piece_vec[2].c_str());
    int y2 = atoi(piece_vec[3].c_str());
    TextLine* tl = new TextLine(x1, y1, x2, y2);

    tllist->push_back(tl);
  }

  in.close();
}
