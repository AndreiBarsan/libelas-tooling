#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <wordexp.h>
#include <zlib.h>
#include <fstream>

#include "config.h"
#include "image.h"
#include "util.h"

namespace fs = std::experimental::filesystem;

bool kitti2klg::EndsWith(const std::string &value, const std::string &ending){
  if (ending.size() > value.size()) {
    return false;
  } else {
    return equal(ending.rbegin(), ending.rend(), value.rbegin());
  }
}

void kitti2klg::ReadTimestampWithNanoseconds(const std::string &input, tm *time, long *nanosecond){
  int year, month, day, hour, minute, second;
  sscanf(input.c_str(), "%d-%d-%d %d:%d:%d.%ld", &year, &month, &day, &hour,
         &minute, &second, nanosecond);
  time->tm_year = year;
  time->tm_mon = month - 1;
  time->tm_mday = day;
  time->tm_hour = hour;
  time->tm_min = minute;
  time->tm_sec = second;
}

fs::path kitti2klg::GetExpandedPath(const std::string &raw_path) {
  wordexp_t expansion_result;
  wordexp(raw_path.c_str(), &expansion_result, 0);
  fs::path foo = expansion_result.we_wordv[0];
  return foo;
}


std::string kitti2klg::Format(const std::string& fmt, ...) {
  // Keeps track of the resulting string size.
  size_t out_size = fmt.size() * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (true) {
    formatted.reset(new char[out_size]);
    std::strcpy(&formatted[0], fmt.c_str());
    va_start(ap, fmt);
    int final_n = vsnprintf(&formatted[0], out_size, fmt.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= out_size) {
      int size_update = final_n - static_cast<int>(out_size) + 1;
      out_size += abs(size_update);
    }
    else {
      break;
    }
  }

  return std::string(formatted.get());
}

std::vector<long> kitti2klg::GetSequenceTimestamps(const fs::path &root) {
  fs::path timestamp_fpath = root;
  timestamp_fpath.append(kitti2klg::KITTI_GRAYSCALE_LEFT_FOLDER);
  timestamp_fpath.append("timestamps.txt");

  std::ifstream in(timestamp_fpath);
  std::vector<long> timestamps;
  std::string chunk;
  while (getline(in, chunk, '\n')) {
    tm time;
    long nanosecond;
    kitti2klg::ReadTimestampWithNanoseconds(chunk, &time, &nanosecond);

    // The format expected by Kintinuous uses microsecond resolution.
    long microsecond = nanosecond / 1000;
    time_t total_seconds = timegm(&time);
    long total_microseconds = total_seconds * 1000 * 1000 + microsecond;

    timestamps.push_back(total_microseconds);
  }

  return timestamps;
}

CvMat* kitti2klg::EncodeJpeg(const cv::Mat &image) {
  // We will use the C-style OpenCV API for consistency with Kintinuous.
  int jpeg_params[] = {CV_IMWRITE_JPEG_QUALITY, 90, 0};

  // Small hack to ensure our dump has RGB channels, even if here, in this
  // tool, we're just dealing with grayscale.
  // TODO(andrei): Use grayscale pairs for depth as before, but pass the
  // actual color left frame to this function.
  cv::Mat raw_mat_col = cv::Mat(image.size(), CV_8U);
  cvtColor(image, raw_mat_col, CV_GRAY2BGR);
  IplImage *raw_ipl_col = new IplImage(raw_mat_col);
  CvMat *encoded = cvEncodeImage(".jpg", raw_ipl_col, jpeg_params);

//    cvReleaseImage(&raw_ipl_col); // realising this here -> segfault.
  return encoded;
}

cv::Mat* kitti2klg::ToCvMat(const image<uchar> &libelas_image) {
  CvSize size = cvSize(libelas_image.width(), libelas_image.height());
  return new cv::Mat(size, CV_8U, libelas_image.data);
}

cv::Mat* kitti2klg::ToCvMat(const image<uint16_t> &libelas_image) {
  CvSize size = cvSize(libelas_image.width(), libelas_image.height());
  return new cv::Mat(size, CV_16U, libelas_image.data);
}

cv::Mat* kitti2klg::ToCvMat(const image<int16_t> &libelas_image) {
  CvSize size = cvSize(libelas_image.width(), libelas_image.height());
  return new cv::Mat(size, CV_16S, libelas_image.data);
}

CvMat* kitti2klg::EncodeJpeg(const image <uchar> &raw_image) {
  cv::Mat *raw_mat = ToCvMat(raw_image);
  CvMat *jpeg = EncodeJpeg(*raw_mat);
  delete raw_mat;
  return jpeg;
}

void kitti2klg::check_compress2(int compress_result) {
  if (compress_result == Z_BUF_ERROR) {
    throw std::runtime_error(
        "zlib Z_BUF_ERROR: Destination buffer too small.");
  } else if (compress_result == Z_MEM_ERROR) {
    throw std::runtime_error("zlib Z_MEM_ERROR; Insufficient memory.");
  } else if (compress_result == Z_STREAM_ERROR) {
    throw std::runtime_error(
        "zlib Z_STREAM_ERROR: Unknown compression level.");
  }
}
