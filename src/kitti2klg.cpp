#include <wordexp.h>

#include <experimental/filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "image.h"
#include "elas.h"

namespace kitti2klg {
  using namespace std;
  namespace fs = std::experimental::filesystem;

  const string KITTI_GRAYSCALE_LEFT_FOLDER  = "image_00";
  const string KITTI_GRAYSCALE_RIGHT_FOLDER = "image_01";
  const string KITTI_COLOR_LEFT_FOLDER      = "image_02";
  const string KITTI_COLOR_RIGHT_FOLDER     = "image_03";

  inline bool ends_with(const string& value, const string& ending) {
    if (ending.size() > value.size()) {
      return false;
    }
    else {
      return equal(ending.rbegin(), ending.rend(), value.rbegin());
    }
  }

  /**
   * @brief Get a list of filenames for KITTI stereo pairs.
   * @param root The folder containing the desired KITTI sequence. The files
   *    must be in the `pgm` format.
   *
   * @return A list of filename pairs containing the corresponding left and
   *    right grayscale image files for every frame.
   */
  vector<pair<fs::path, fs::path>> get_kitti_stereo_pair_paths(const fs::path &root) {

    cout << "Current path: " << fs::current_path() << endl;
    fs::path left_dir = root;
    left_dir.append(KITTI_GRAYSCALE_LEFT_FOLDER);
    left_dir.append("data");

    if (ends_with("derp herp", "erp")) {
      cout << "OK!" << endl;
    }
    else {
      cout  << "BAD!" << endl;
    }

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
      if (ends_with(left_it->path().string(), ".pgm") &&
          ends_with(right_it->path().string(), ".pgm")) {
        result.emplace_back(left_it->path(), right_it->path());
      }
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

  void compute_depth(const pair<fs::path, fs::path>& pair) {
    // Heavily based on the demo program which ships with libelas.
    cout << "Processing " << pair.first << ", " << pair.second << endl;

    image<uchar> *left, *right;
    left = loadPGM(pair.first.string().c_str());
    right = loadPGM(pair.second.string().c_str());

    // check for correct size
    if (left->width()<=0 || left->height() <=0 || right->width()<=0 || right->height() <=0 ||
        left->width()!=right->width() || left->height()!=right->height()) {
      cout << "ERROR: Images must be of same size, but" << endl;
      cout << "       left: " << left->width() <<  " x " << left->height() <<
           ", right: " << right->width() <<  " x " << right->height() << endl;
      delete left;
      delete right;
      return;
    }

    // get image width and height
    int32_t width  = left->width();
    int32_t height = left->height();

    // allocate memory for disparity image
    const int32_t dims[3] = {width, height, width}; // bytes per line = width
    float* D1_data = (float*)malloc(width*height*sizeof(float));
    // TODO(andrei): Could we just not allocate this?
    float* D2_data = (float*)malloc(width*height*sizeof(float));

    // process
    Elas::parameters param;
    param.postprocess_only_left = true;
    Elas elas(param);
    elas.process(left->data, right->data, D1_data, D2_data, dims);

    // find maximum disparity for scaling output disparity images to [0..255]
    float disp_max = 0;
    for (int32_t i=0; i<width*height; i++) {
      if (D1_data[i]>disp_max) disp_max = D1_data[i];
      if (D2_data[i]>disp_max) disp_max = D2_data[i];
    }

    // TOOD(andrei): Move this to own rendering function.
    // copy float to uchar
    image<uchar> *D1 = new image<uchar>(width,height);
    image<uchar> *D2 = new image<uchar>(width,height);
    for (int32_t i=0; i<width*height; i++) {
      D1->data[i] = (uint8_t)max(255.0*D1_data[i]/disp_max,0.0);
      D2->data[i] = (uint8_t)max(255.0*D2_data[i]/disp_max,0.0);
    }

    // Save disparity images in sibling to 'image_00' folder.
    fs::path output = pair.first.parent_path().parent_path().parent_path();
    // TODO(andrei): Create this folder automatically if needed.
    output.append("depth");
    output.append(pair.first.stem().string());
    output.concat(".depth.pgm");
    cout << "Dumping depth map in [" << output << "]." << endl;
    savePGM(D1, output.string().c_str());

    // free memory
    delete left;
    delete right;
    delete D1;
    delete D2;
    free(D1_data);
    free(D2_data);
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
    std::cout << "No '*.pgm' files found in the KITTI root folder ["
              << kitti_seq_path << "]." << std::endl;
    return 1;
  }
  else {
    std::cout << "Found " << res.size() << " stereo pairs to process." << std::endl;

    for (const std::pair<fs::path, fs::path>& stereo_pair_fpaths: res) {
      kitti2klg::compute_depth(stereo_pair_fpaths);

      std::cout << stereo_pair_fpaths.first << ", "
                << stereo_pair_fpaths.second << std::endl;
    }
  }


  return 0;
}
