#include <wordexp.h>

#include <experimental/filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace kitti2klg {
  using namespace std;
  namespace fs = std::experimental::filesystem;

  const string KITTI_GRAYSCALE_LEFT_FOLDER  = "image_00";
  const string KITTI_GRAYSCALE_RIGHT_FOLDER = "image_01";
  const string KITTI_COLOR_LEFT_FOLDER      = "image_02";
  const string KITTI_COLOR_RIGHT_FOLDER     = "image_03";

  /**
   * @brief Get a list of filenames for KITTI stereo pairs.
   * @param root The folder containing the desired KITTI sequence.
   * @return A list of filename pairs containing the corresponding left and
   *    right grayscale image files for every frame.
   */
  vector<pair<fs::path, fs::path>> get_kitti_stereo_pair_paths(const fs::path &root) {

    cout << "Current path: " << fs::current_path() << endl;
    fs::path left_dir = root;
    left_dir.append(KITTI_GRAYSCALE_LEFT_FOLDER);
    left_dir.append("data");

    /*
     * Folder structure for a stereo pair:
     *  date/date_drive_id/image_00/data/*.png for left, grayscale
     *  date/date_drive_id/image_01/data/*.png for left, grayscale
     */
    fs::path right_dir = root;
    right_dir.append(KITTI_GRAYSCALE_RIGHT_FOLDER);
    right_dir.append("data");

    cout << "Left dir: " << left_dir << endl << "Right dir: " << right_dir << endl;
    // TODO(andrei): Consider asserting both dirs have the same number of files.

    vector<pair<fs::path, fs::path>> result;

    // Iterate through the left-image and the right-image directories
    // simultaneously, grabbing the file names along the way.
    for(fs::directory_iterator left_it = fs::begin(fs::directory_iterator(left_dir)),
        right_it = fs::begin(fs::directory_iterator(right_dir));
        left_it != fs::end(fs::directory_iterator(left_dir)) && right_it != fs::end(fs::directory_iterator(right_dir));
        ++left_it, ++right_it) {
      result.emplace_back(left_it->path(), right_it->path());
    }

    return result;
  };

  fs::path get_expanded_path(const string& raw_path) {
    // We C now!
    wordexp_t expansion_result;
    wordexp(raw_path.c_str(), &expansion_result, 0);
    fs::path foo = expansion_result.we_wordv[0];
    return foo;
  };

  void load_jpg(const fs::path& path) {
    cout << "Will load jpg from path " << path << endl;
  }

}

int main() {
  namespace fs = std::experimental::filesystem;

  std::cout << "Loading KITTI pairs from folder [] and outputting "
      "ElasticFusion/Kintinuous-friendly *.klg file." << std::endl;

  // TODO(andrei): Docs + bash script for first converting pngs to pgms.


  fs::path kitti_root = kitti2klg::get_expanded_path("~/datasets/kitti");
  fs::path kitti_seq_path = kitti_root / "2011_09_26" / "2011_09_26_drive_0095_sync";
  auto res = kitti2klg::get_kitti_stereo_pair_paths(kitti_seq_path);
  if (res.size() == 0) {
    std::cout << "No results (empty vector)." << std::endl;
    return 1;
  }
  else {
    for (const std::pair<std::string, std::string>& dude: res) {
      std::cout << dude.first << ", " << dude.second << std::endl;
    }

    kitti2klg::load_jpg(res[0].first);
  }


  return 0;
}
