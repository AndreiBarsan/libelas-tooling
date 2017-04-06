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
  // We C now!
  wordexp_t expansion_result;
  wordexp(raw_path.c_str(), &expansion_result, 0);
  fs::path foo = expansion_result.we_wordv[0];
  return foo;
}
