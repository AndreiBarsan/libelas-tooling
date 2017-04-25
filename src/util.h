#ifndef LIBELAS_UTIL_H
#define LIBELAS_UTIL_H

#include <experimental/filesystem>
#include <string>
#include <cstdarg>
#include <cstring>

namespace kitti2klg {
  namespace fs = std::experimental::filesystem;

  /// Checks if 'value' ends with the specified 'ending'.
  bool EndsWith(const std::string& value, const std::string& ending);

  /// Reads a full timestamp with nanosecond resolution seconds, such as
  /// "2011-09-26 15:20:11.552379904".
  /// Populates the standard C++ time object, plus an additional long containing
  /// the nanoseconds (since the standard `tm` object only has second-level
  /// accuracy).
  ///
  /// \param input Time string to parse.
  /// \param time Second-resolution C++ time object (out parameter).
  /// \param nanosecond Additional nanoseconds (out parameter).
  void ReadTimestampWithNanoseconds(
      const std::string &input,
      tm *time,
      long *nanosecond
  );

  /// \brief Performs shell-like expansion.
  /// \param raw_path Unexpanded UNIX path, e.g., `~/work` or `${HOME}/work`.
  /// \return Typesafe C++ object containing the expanded path.
  fs::path GetExpandedPath(const std::string& raw_path);

  /// \brief Convenient string formatting utility.
  /// Originally from StackOverflow: https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
  std::string Format(const std::string& fmt, ...);

  /// NOTE: This only looks at the timestamps associated with the left
  /// greyscale images. There are also timestamps associated with each of the
  /// other cameras, and they are not exactly identitcal. Nevertheless, this
  /// approach should be good enough for the current application.
  ///
  /// \returns A vector of UTC timestamps at MICROSECOND resolution, necessary
  ///   for the custom Kintinuous `.klg` format.
  std::vector<long> GetSequenceTimestamps(const fs::path &root);

  /// \brief Encodes a raw image as JPEG, for use in the 'klg' log.
  /// \return A 1D CvMat pointer to the compressed byte representation. The
  /// caller takes ownership of this memory.
  ///
  /// Converts the image to BGR format before encoding it.
  ///
  /// \see EncodeJpeg(const image<uchar> * const)
  CvMat *EncodeJpeg(const cv::Mat &image);

  /// \brief Converts a grayscale libelas image to an OpenCV one.
  /// \return The same image as an OpenCV Mat. The caller takes ownership of
  /// this memory.
  /// TODO-LOW(andrei): Make this function into a template to also support
  /// RGB images and other image depths, reducing code duplication.
  /// TODO-LOW(andrei): Support target buffer to allow memory reuse.
  cv::Mat *ToCvMat(const image<uchar> &libelas_image);

  cv::Mat *ToCvMat(const image<uint16_t> &libelas_image);

  /// \brief Encodes a raw image as JPEG, for use in the 'klg' log.
  /// \return A 1D CvMat pointer to the compressed byte representation. The
  /// caller takes ownership of this memory. CvMat is used for compatibility
  /// reasons with the classic Kintinuous log file format.
  CvMat *EncodeJpeg(const image <uchar> &raw_image);

  /// Checks the return result of zlib's `compress2` function, throwing an
  /// error if compression failed.
  void check_compress2(int compress_result);
}

#endif // LIBELAS_UTIL_H
