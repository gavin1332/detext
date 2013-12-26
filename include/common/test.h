#ifndef COMMON_TEST_H_
#define COMMON_TEST_H_

#include <vector>
#include <list>
#include <sstream>
#include <iostream>

#include <opencv2/core/core.hpp>

namespace dtxt {

class TestUtils {
 public:
  // Print some content into standard output
  template<typename T>
  static std::ostream& Print(const T& value) {
    return std::cout << value << std::endl;
  }

  // Print some key-value pair into standard output
  template<typename T>
  static std::ostream& Print(const std::string& key, const T& value) {
    return std::cout << key << ": " << value << std::endl;
  }

  // LOG
  template<typename T>
  static void Log(const std::string& key, const T& value) {
    extern bool PRINT_LOG_;
    if (PRINT_LOG_)
      std::cout << "[INFO] " << key << ": " << value << std::endl;
  }

  // LOG
  template<typename T>
  static void Log(const T& value) {
    extern bool PRINT_LOG_;
    if (PRINT_LOG_)
      std::cout << "[INFO] " << value << std::endl;
  }

  static std::ostream& PrintRect(const cv::Rect& rect) {
    return std::cout << "Rect(x, y, w, h): (" << rect.x << "," << rect.y << ","
                     << rect.width << "," << rect.height << ")" << std::endl;
  }

  // @see
  // bool ShowImage(const char* win_name, const cv::Mat& img, int wait_time);
  static bool ShowImage(const cv::Mat& img) {
    return ShowImage("Untitled", img, 0);
  }

  // Show a single image contained in Mat with specified window name.
  static bool ShowImage(const char* win_name, const cv::Mat& img) {
    return ShowImage(win_name, img, 0);
  }

  // Show a single image contained in Mat with specified window name and
  // waiting time.
  //
  // This function supports some convenient operations when window is active:
  // press "n": notify the caller with a "true" returned value;
  // press "s": save the current image as a BMP file with a random filename;
  // press "+": zoom in;
  // press "-": zoom out;
  // press "0": resume size;
  // press ESC: exit program, and report the result when calculating;
  // press space bar or ENTER: close current window.
  // click mouse on the image: print the coordinate of the click position.
  static bool ShowImage(const char* win_name, const cv::Mat& img,
                        int wait_time);

  static const int MAX_IMAGE_NUM = 12;

  // This function illustrates how to show several image in a single window
  //
  // @param const char* win_name: The name of the window to be showed
  // @param const cv::Mat*: An image array stored in cv::Mat
  // @param const int N: The number of images
  // @param int wait_time: The time(ms) to wait if no key is pressed
  //
  // It will keep the width/height ratio of original images, And can show upto
  // 12 images in a single window.
  //
  // Take care of the type of arguments, which should be of Mat with CV_8UC1
  // or CV_8UC3. Otherwise, Error may occur.
  static bool ShowManyImages(const char* win_name, const cv::Mat* img,
                             const int N, int wait_time);

  // This function illustrates how to show several image in a single window
  //
  // @param const char* win_name: The name of the window to be showed
  // @param int argc: The number of images to be showed
  // @param ...: Mat pointers, which contains the images
  //
  // This function can be called like this:
  // showManyImages("Images", 5, img2, img2, img3, img4, img5);
  //
  // It will keep the width/height ratio of original images, And can show upto
  // 12 images in a single window.
  //
  // Take care of the type of arguments, which should be of Mat with CV_8UC1
  // or CV_8UC3. Otherwise, Error may occur.
  static bool ShowManyImages(const char* win_name, int argc, ...);

  // This function pack many images into a single one
  //
  // @param const cv::Mat*: An image array stored in cv::Mat
  // @param const int N: The number of images
  //
  // This function can be called like this:
  // PackManyImages("Images", 5, img2, img2, img3, img4, img5);
  //
  // It will keep the width/height ratio of original images, And can show upto
  // 12 images in a single window.
  //
  // Take care of the type of arguments, which should be of Mat with CV_8UC1
  // or CV_8UC3. Otherwise, Error may occur.
  static cv::Mat PackManyImages(const cv::Mat* imgs, const int N);

  // Show gradient image
  //
  // @param const cv::Mat* grad: grad[0] - magnitudes, grad[1] - directions in [0, 1]
  // @param const float mag_shield: magnitude shield threshold
  //
  static void ShowGradient(const cv::Mat* grad, float mag_shield,
                           bool white_bg = true);

  // @see void ShowNormedHistogram(const float* data, int len, int zoom)
  static void ShowHistogram(const int* data, int len, int zoom);

  // Show normalized histogram.
  //
  // @param const float* data: An normalized float data, the sum of which is 1
  // @param const int len: The length of the data
  // @param int zoom: zoom scale. Positive to enlarge and negative to shrink.
  //
  // Notice: if zoom value is negative, neighbor bins will assemble into one.
  static void ShowNormedHistogram(const float* data, int len, int zoom);

  // Show rectangle in the given image
  static bool ShowRect(const cv::Mat& gray_rgb, const cv::Rect& rect,
                       cv::Scalar bgr = cv::Scalar(255, 255, 255),
                       int thickness = 1);

  // Show rectangles in the given image
  static bool ShowRects(const cv::Mat& gray_rgb,
                        const std::vector<cv::Rect>& rect_vec, cv::Scalar bgr =
                            cv::Scalar(255, 255, 255));

  // Show line in the given image
  static bool ShowLine(const cv::Mat& gray_rgb, cv::Point p1, cv::Point p2,
                       cv::Scalar bgr = cv::Scalar(255, 255, 255));

  // Show point in the given image
  static bool ShowPoint(const cv::Mat& gray_rgb, cv::Point point,
                        cv::Scalar bgr = cv::Scalar(255, 255, 255));

  // Show points in the given image
  static bool ShowPoints(const cv::Mat& gray_rgb,
                         std::vector<cv::Point> point_vec, cv::Scalar bgr =
                             cv::Scalar(255, 255, 255));

  // Crop image by user interaction
  static bool UserDrawRect(const cv::Mat& gray_rgb, cv::Rect* rect);

 private:
  typedef struct crop_param {
    cv::Point start;
    cv::Point end;
    bool drawing;
    bool finished;
  } CropParam;

  static bool UserInteraction(const char* win_name, const cv::Mat& img,
                              int wait_time);

  // Determine the size of the image and the number of rows/cols
  // from number of arguments in ShowManyImage function
  static void CalcLayout(int argc, int* rows, int* cols, int* inner_size);

  static cv::Mat CreateCanvas(int rows, int cols, int inner_size,
                              int inner_space);

  static void Resize2Canvas(const cv::Mat& img, cv::Mat* canvas, cv::Point pos,
                            int inner_size);

  static void CmnMouseCallback(int event, int x, int y, int flags, void *param);

  static void CropMouseCallback(int event, int x, int y, int flags,
                                void *param);
};

}

#endif
