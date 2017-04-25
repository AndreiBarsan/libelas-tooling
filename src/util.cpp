#include <wordexp.h>
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
