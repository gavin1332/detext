#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace dtxt {

class CmnUtils {
 public:
  static std::string Trim(const std::string& input);

  static void Split(const std::string& input, char delim,
                    std::vector<std::string>* output);

  // Retrieve all file names with a named suffix in the specified folder
  static void RetrieveFilenames(const std::string& dir_name,
                                const std::string& suffix,
                                std::vector<std::string>* filename_list,
                                bool ignore_case);

 private:
  static bool IsBlank(const char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
  }

};

class MathUtils {
 public:
  static const double kPI;

  static bool IsAbsErr0(float f) {
    return CmpInAbsErr(f, 0) == 0;
  }

  static bool IsAbsErr0(double d) {
    return CmpInAbsErr(d, 0) == 0;
  }

  static int CmpInAbsErr(float a, float b) {
    if (fabs(a - b) <= 1e-6) {
      return 0;
    }

    return (a < b) ? -1 : 1;
  }

  static int CmpInAbsErr(double a, double b) {
    if (fabs(a - b) <= DBL_EPSILON) {
      return 0;
    }

    return (a < b) ? -1 : 1;
  }

  template<typename T>
  static void MeanAndStdDev(const std::vector<T>& data_list, double* mean,
                            double* std_dev);

  // Check whether the input two erect rectangles is intersected, and if true,
  // assign the parameter "intersect" with intersected rectangle
  static bool IsRectIntersected(const cv::Rect& a, const cv::Rect& b,
                                cv::Rect* intersect = NULL);

  // Calculate the minimum bounding box
  static cv::Rect MinRectBoundingBox(const cv::Rect& a, const cv::Rect& b) {
    return cv::Rect(
        std::min(a.x, b.x), std::min(a.y, b.y),
        std::max(a.x + a.width, b.x + b.width) - std::min(a.x, b.x),
        std::max(a.y + a.height, b.y + b.height) - std::min(a.y, b.y));
  }

};

template<typename T>
void MathUtils::MeanAndStdDev(const std::vector<T>& data_list, double* mean,
                              double* std_dev) {
  size_t N = data_list.size();

  double sum = 0;
  typename std::vector<T>::const_iterator it = data_list.begin();
  for (; it != data_list.end(); ++it) {
    sum += *it;
  }
  *mean = sum / N;

  sum = 0;
  it = data_list.begin();
  for (; it != data_list.end(); ++it) {
    sum += (*it - *mean) * (*it - *mean);
  }
  *std_dev = sqrt(sum / N);
}

}
;

#endif
