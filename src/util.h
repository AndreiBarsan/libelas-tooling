#ifndef LIBELAS_UTIL_H
#define LIBELAS_UTIL_H

#include <experimental/filesystem>
#include <string>

namespace kitti2klg {
  namespace fs = std::experimental::filesystem;

  inline bool EndsWith(const std::string& value, const std::string& ending);

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
}

#endif //LIBELAS_UTIL_H
