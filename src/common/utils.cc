#include "common/utils.h"

#include <sys/stat.h>
#include <dirent.h>

#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace dtxt;

string CmnUtils::Trim(const string& input) {
  if (input.length() == 0) {
    return string();
  }

  string::const_iterator it = input.begin();
  string::const_iterator end = input.end();
  for (; it != end && IsBlank(*it); ++it) {
  }
  if (it == end) {
    return string();
  }
  string::const_iterator start = it;

  for (it = input.end() - 1; it != start && IsBlank(*it); --it) {
  }

  return string(start, it + 1);
}

void CmnUtils::Split(const string& input, char delim, vector<string>* output) {
  stringstream ss(input);
  std::string item;
  while (getline(ss, item, delim)) {
    output->push_back(item);
  }
}

void CmnUtils::RetrieveFilenames(const string& dir_name, const string& suffix,
                                 vector<string>* filename_list,
                                 bool ignore_case) {
  string complete_dir = dir_name;
  if (complete_dir.at(complete_dir.length() - 1) != '/') {
    complete_dir += '/';
  }

  DIR* dir = opendir(complete_dir.c_str());
  if (dir == NULL) {
    cerr << "Cannot open directory: " << complete_dir << endl;
    return;
  }

  struct stat file_stat;
  while (true) {
    struct dirent* dirent = readdir(dir);
    if (dirent == NULL) {
      break;
    }

    string path = complete_dir + dirent->d_name;
    int res = stat(path.c_str(), &file_stat);
    assert(res == 0);

    if (S_ISREG(file_stat.st_mode)) {  // regular file
      string filename(dirent->d_name);

      string temp_suffix = filename.substr(filename.length() - suffix.length(),
                                           suffix.length());

      string suffix_target = suffix;
      if (ignore_case) {
        std::transform(temp_suffix.begin(), temp_suffix.end(),
                       temp_suffix.begin(), ::tolower);

        std::transform(suffix_target.begin(), suffix_target.end(),
                       suffix_target.begin(), ::tolower);
      }

      if (temp_suffix.compare(suffix_target) == 0) {
        filename_list->push_back(filename);
      }
    }
  }

  closedir(dir);
}

bool MathUtils::IsRectIntersected(const Rect& a, const Rect& b,
                                  Rect* intersect) {
  int x = std::max(a.x, b.x);
  int y = std::max(a.y, b.y);
  int width = std::min(a.x + a.width, b.x + b.width) - x;
  int height = std::min(a.y + a.height, b.y + b.height) - y;

  if (intersect != NULL) {
    *intersect = Rect(x, y, width, height);
  }

  return width > 0;
}
