
#include <experimental/filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// WARNING: There is likey something rotten with this code. I have NO idea
// what it is, as of May 7th 2017, BUT the depth maps it produces seem to
// give InfiniTAM indigestion, and I have no idea why.

// TODO(andrei): If you decide to stick with gflags, mention them as a
// dependency in the README. Or, better yet, add them as a git submodule, and
// depend on them elegnatly from cmake.
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <zlib.h>
#include <execinfo.h>
#include <csignal>
#include <limits>
#include <iomanip>
#include <sys/time.h>

#include "config.h"
#include "image.h"
#include "elas.h"
#include "util.h"

// Support for thread pools in C++ (stl version == no Boost).
#include "ctpl_stl.h"

namespace kitti2klg {
using namespace std;
namespace fs = std::experimental::filesystem;

using stereo_fpath_pair = std::pair<fs::path, fs::path>;
using img_ptr = shared_ptr<image<uchar>>;
using stereo_image_pair = std::pair<img_ptr, img_ptr>;

// Command-line argument definitions, using the elegant `gflags` library.
DEFINE_bool(infinitam, false, "Whether to generate InfiniTAM-style dump folders");
//  "2011_09_26_drive_0095_sync" is a good demo sequence.
DEFINE_string(kitti_root, "", "Location of the input KITTI sequence.");
DEFINE_string(output, "out_log.klg", "Output file name (when using "
    "Kintinuous logger format, folder when using InfiniTAM format).");
DEFINE_int32(process_frames, -1, "Number of frames to process. Set to -1 "
    "to process the entire sequence.");
DEFINE_bool(use_color, true, "Whether add color information to the "
    "resulting dump. If disabled, grayscale info is used instead.");

DEFINE_string(calib_file, "", "Calibration file for the sequence, specified as a path "
    "relative to 'kitti_root' parameter.");
DEFINE_double(max_depth_meters, 30.0, "Depth values larger than this are discarded.");
DEFINE_double(min_depth_meters, 0.5, "Depth values smaller than this are discarded.");
// e.g., for KITTI this is 0.537150654273.
DEFINE_double(baseline_meters, -1.0, "The baseline length of the stereo rig.");

/// \brief Stores information about a frame of visual data.
struct StereoFrameFpaths {
  fs::path left_gray_fpath;
  fs::path right_gray_fpath;
  fs::path left_color_fpath;
  fs::path right_color_fpath;

  StereoFrameFpaths(const fs::path &left_gray_fpath,
                    const fs::path &right_gray_fpath,
                    const fs::path &left_color_fpath,
                    const fs::path &right_color_fpath) : left_gray_fpath(
      left_gray_fpath), right_gray_fpath(right_gray_fpath), left_color_fpath(
      left_color_fpath), right_color_fpath(right_color_fpath) {}

  // TODO-LOW(andrei): More data, if necessary, such as Velodyne points,
  // IMU info, etc.
};

/// \brief Computes the full paths of the files in `dir`, having `extension`.
vector<fs::path> ListDir(const fs::path &dir, const string &extension) {
  auto dir_it = fs::directory_iterator(dir);
  vector<fs::path> fileList;
  for (auto it = fs::begin(dir_it); it != fs::end(dir_it); ++it) {
    if (kitti2klg::EndsWith(it->path().string(), extension)) {
      fileList.push_back(it->path());
    }
  }
  return fileList;
}

/// \brief Turns to vectors into a single vector of pairs.
/// \note The resulting vector is as long as the shortest of the inputs.
template<typename T, typename U>
vector<std::pair<T, U>> Zip(std::vector<T> left, std::vector<U> right) {
  vector<std::pair<T, U>> result;
  for (auto it_l = left.cbegin(), it_r = right.cbegin();
       it_l != left.cend(), it_r != right.cend();
       ++it_l, ++it_r) {
    result.emplace_back(*it_l, *it_r);
  }
  return result;
};

int64_t GetTimeMs() {
  struct timeval time;
  gettimeofday(&time, NULL);
  int64_t time_ms = time.tv_sec * 1000 + time.tv_usec / 1000;
  return time_ms;
}

/**
 * @brief Get a list of filenames for KITTI stereo pairs.
 * @param sequence_root The folder containing the desired KITTI sequence.
 * @param image_extension The extension of the image type to look for.
 *
 * @return A list of filename pairs containing the corresponding left and
 *    right grayscale image files for every frame.
 */
vector<StereoFrameFpaths> GetKittiStereoPairPaths(
    const fs::path &sequence_root,
    const string image_extension = ".png"
) {
  // Libelas only cares about intensity, so we use the grayscale images for
  // the depth, but feed the colored left frame to the downstream SLAM
  // system to get colored maps.
  // So why can't we just load the color images and convert them to
  // grayscale before giving them to libelas?
  // Great question, Billy! It's because the gray and color images from the
  // KITTI dataset have been taken using different cameras. And the
  // grayscale cameras actually have a better dynamic range than the color
  // ones. Otherwise, there wouldn't even be any point to using both
  // grayscale and color cameras!
  fs::path left_gray_dir, right_gray_dir;
  fs::path left_color_dir, right_color_dir;
  if (fs::exists(sequence_root / KITTI_GRAYSCALE_LEFT_FOLDER)) {
    // Looks like a regular KITTI sequence.
    left_gray_dir = sequence_root / KITTI_GRAYSCALE_LEFT_FOLDER / "data";
    right_gray_dir = sequence_root / KITTI_GRAYSCALE_RIGHT_FOLDER / "data";
    left_color_dir = sequence_root / KITTI_COLOR_LEFT_FOLDER / "data";
    right_color_dir = sequence_root / KITTI_COLOR_RIGHT_FOLDER / "data";
  } else if (fs::exists(sequence_root / "image_0")) {
    // Looks like a KITTI odometry benchmark sequence.
    left_gray_dir = sequence_root / "image_0";
    right_gray_dir = sequence_root / "image_1";
    left_color_dir = sequence_root / "image_2";
    right_color_dir = sequence_root / "image_3";
  } else {
    // Note: In the future, we could support more stereo sequences, like those
    // from the Cityscapes, Karlsruhe, etc. datasets.
    throw runtime_error(Format("Unknown type of sequence in folder [%s].",
                               sequence_root.string()));
  }

  vector<fs::path> left_gray_fpaths = ListDir(left_gray_dir, image_extension);
  vector<fs::path> right_gray_fpaths = ListDir(right_gray_dir, image_extension);
  // TODO(andrei): Don't try to load color if the folders aren't there (for
  // the odometry dataset, color data can be downloaded separately).
  vector<fs::path> left_color_fpaths = ListDir(left_color_dir, image_extension);
  vector<fs::path> right_color_fpaths = ListDir(right_color_dir, image_extension);

  if (left_gray_fpaths.size() != right_gray_fpaths.size() ||
      left_gray_fpaths.size() != left_color_fpaths.size() ||
      left_gray_fpaths.size() != right_color_fpaths.size()
      ) {
    throw runtime_error("Different frame counts in the stereo folders.");
  }

  vector<StereoFrameFpaths> result;
  for (size_t i = 0; i < left_gray_fpaths.size(); ++i) {
    result.push_back(StereoFrameFpaths(
        left_gray_fpaths[i],
        right_gray_fpaths[i],
        left_color_fpaths[i],
        right_color_fpaths[i]
    ));
  }

  // Explicitly sort the paths so that they're in ascending order, since the
  // directory iterator does not guarantee it, and it's useful to iterate
  // through the frames in order when working on subsets of the data.
  auto compare_path_pairs = [](const StereoFrameFpaths &stereo_pair_A,
                               const StereoFrameFpaths &stereo_pair_B) {
    return stereo_pair_A.left_gray_fpath.filename().string() <
        stereo_pair_B.left_gray_fpath.filename().string();
  };
  sort(result.begin(), result.end(), compare_path_pairs);

  return result;
}

/// Loads an image. Currently only PGM and PNG are supported. The image is
/// loaded as grayscale, since that's all libelas cares about.
///
/// \returns The loaded uchar image.
img_ptr LoadImage(const fs::path &image_fpath) {
  if (!image_fpath.has_extension()) {
    throw runtime_error(Format(
        "Cannot load images without extensions, as format sniffing is "
            "currently not supported. Problematic path: %s", image_fpath.string()));
  }

  if (image_fpath.extension() == ".png") {
    cv::Mat img = cv::imread(image_fpath, CV_LOAD_IMAGE_ANYDEPTH);
    return make_shared<image<uchar>>(img.cols, img.rows, img.data);
  } else if (image_fpath.extension() == ".pgm") {
    return shared_ptr<image<uchar>>(loadPGM(image_fpath.string().c_str()));
  } else {
    throw runtime_error(Format(
        "Unsupported image format: %s", image_fpath.extension()));
  }
}

/// \brief Loads the gray images from a stereo frame.
/// \note We don't need the color information for the depth when using libelas.
unique_ptr<stereo_image_pair> LoadStereoPair(const StereoFrameFpaths &pair_fpaths) {
  auto result = make_unique<stereo_image_pair>(
      LoadImage(pair_fpaths.left_gray_fpath), LoadImage(pair_fpaths.right_gray_fpath)
  );
  auto left = result->first;
  auto right = result->second;

  if (left->width() <= 0 || left->height() <= 0 || right->width() <= 0 ||
      right->height() <= 0 || left->width() != right->width() ||
      left->height() != right->height()) {
    stringstream err;
    err << "ERROR: Images must be of same size, but" << endl
        << "       left: " << left->width() << " x " << left->height()
        << ", right: " << right->width() << " x " << right->height() << endl;

    // This also destroys the image pair unique_ptr.
    throw runtime_error(err.str());
  }

  return result;
};

/// \brief Computes a metric depth value from the supplied disparity and calibration parameters.
uint16_t DepthFromDisparity(double disparity_px, double baseline_m, double focal_length_px,
                            double min_depth_m, double max_depth_m
) {
  const uint16_t kInvalidInfinitamDepth = numeric_limits<uint16_t>::max();
  const float kMetersToCentimeters = 100.0f;
  const float kMetersToMilimeters = kMetersToCentimeters * 10.0f;
  // Discard depth measurements too close to, or too far from the camera, as they tend to be noisy.
  double lower_threshold_mm = min_depth_m * kMetersToMilimeters;
  double upper_threshold_mm = max_depth_m * kMetersToMilimeters;

  if (upper_threshold_mm >= kInvalidInfinitamDepth) {
    throw runtime_error("Upper depth threshold larger than the max value which can be stored in "
                            "the depth map.");
  }

  if (disparity_px == kInvalidDepth) {
    // If libelas flags this as an invalid depth measurement, we want to propagate that to the
    // underlying SLAM system.
    return kInvalidInfinitamDepth;
  } else {
    // Use the classic formula: Z = (b * f) / disparity.
    double depth_m_f = (baseline_m * focal_length_px) / disparity_px;
    double depth_mm_f = depth_m_f * kMetersToMilimeters;
    int32_t depth_mm_u32 = static_cast<int32_t>(depth_mm_f);
    if (depth_mm_u32 < lower_threshold_mm || depth_mm_u32 > upper_threshold_mm) {
      return kInvalidInfinitamDepth;
    }

    return static_cast<uint16_t>(depth_mm_u32);
  }
}

/// \brief Uses libelas to compute the disparity map from a stereo pair.
/// Only computes the map in the left camera's frame.
/// \param depth_out The preallocated depth map object, to be populated by
/// this function.
void ComputeDepthElas(
    const image<uchar> &left,
    const image<uchar> &right,
    const double baseline_m,
    const double focal_length_px,
    const double min_depth_m,
    const double max_depth_m,
    image<uint16_t> *depth_out
) {
  // Heavily based on the demo program which ships with libelas.
  int32_t width = left.width();
  int32_t height = left.height();

  // allocate memory for disparity image
  const int32_t dims[3] = {width, height, width}; // bytes per line = width
  float *D1_data = (float *) malloc(width * height * sizeof(float));
  // 'D2_data' is necessary inside libelas, but not used by our code.
  float *D2_data = (float *) malloc(width * height * sizeof(float));

  // process
  Elas::parameters params(Elas::ROBOTICS);   // Use the default config.
  // Enabling corners causes artifacts in the corners of the frames, leading to downwards-facing
  // areas in the 3D reconstructions.
  params.add_corners = 0;
  params.match_texture = 1;
  // The bigger this is, the bigger the gaps we interpolate over.
  // The default is 3. 7 leads to decent results. 11 is OK and seems
  // slightly better, with a slightly lower cost in terms of memory.
  // 31 doesn't change much, but the depth maps do look a bit oversmoothed.
  // Around 100 the artifacts become quite large. Setting it to a conservative '1' still produces
  // artifacts in the sky and around trees.
  params.ipol_gap_width = 21;
  params.postprocess_only_left = true;

  Elas elas(params);
  elas.process(left.data, right.data, D1_data, D2_data, dims);

  // Convert the float disparity map to 16-bit unsigned int depth map,
  // expressed in centimeters.
  for (int32_t i = 0; i < width * height; i++) {
    // Output 16-bit depth values, in the Kinect style.
    depth_out->data[i] = DepthFromDisparity(D1_data[i], baseline_m, focal_length_px,
                                            min_depth_m, max_depth_m);
  }

  free(D1_data);
  free(D2_data);
}

/// \brief Computes the depth map from a stereo pair.
/// Only computes the map in the left camera's frame.
/// \param depth_out The pre-allocated depth map object, to be populated by this function.
void ComputeDepth(
    const image<uchar> &left,
    const image<uchar> &right,
    const double baseline_m,
    const double focal_length_px,
    const double min_depth_m,
    const double max_depth_m,
    image<uint16_t> *depth_out
) {
  // Note: can add support for other libraries here, such as OpenCV's
  // built-in SGM, or something else.
  return ComputeDepthElas(left, right, baseline_m, focal_length_px, min_depth_m, max_depth_m, depth_out);
}


void ProcessInfinitamFrame(int idx,
                           const experimental::filesystem::path &output_path,
                           const bool use_color,
                           double baseline_m,
                           double focal_length_px,
                           double min_depth_m,
                           double max_depth_m,
                           const StereoFrameFpaths &pair_fpaths
) {
  auto img_pair = LoadStereoPair(pair_fpaths);

  int32_t width = img_pair->first->width();
  int32_t height = img_pair->second->height();

// TODO(andrei): Just make sure sizes are consistent throughout the
// sequence, otherwise this check is just a PITA.
//      if(width != kKittiFrameWidth || height != kKittiFrameHeight) {
//        throw runtime_error(Format(
//            "Unexpected image dimensions encountered! Was assuming standard "
//            "KITTI frame dimensions of %d x %d.", kKittiFrameWidth, kKittiFrameHeight));
//      }

  cv::Mat left_frame_cv_col; //(left_frame_cv.size(), CV_8U);
  if (use_color) {
    left_frame_cv_col = cv::imread(pair_fpaths.left_color_fpath.string());
  } else {
    // Process the intensity image, converting it to RGB.
    cv::Mat *temp = ToCvMat(*(img_pair->first));
    cv::Mat left_frame_cv = cv::Mat(*temp);
    delete temp;
    cvtColor(left_frame_cv, left_frame_cv_col, CV_GRAY2BGR);
  }

  auto depth = make_shared<image<uint16_t>>(width, height);
  int64_t disp_start = GetTimeMs();
  ComputeDepth(*(img_pair->first), *(img_pair->second),
               baseline_m, focal_length_px, min_depth_m, max_depth_m,
               depth.get());
  int64_t disp_time = GetTimeMs() - disp_start;
  if (idx % 17 == 0) {
    cout << "Processing " << pair_fpaths.left_gray_fpath.filename() << ". "
         << "Took " << disp_time << "ms." << endl;
  }

  cv::Mat *depth_cv = ToCvMat(*depth);

  ostringstream depth_fname;
  depth_fname << setfill('0') << setw(4) << idx << ".pgm";
  stringstream color_fname;
  color_fname << setfill('0') << setw(4) << idx << ".ppm";

  fs::path depth_fpath = output_path / "Frames" / depth_fname.str();
  fs::path color_fpath = output_path / "Frames" / color_fname.str();

  imwrite(depth_fpath, *depth_cv);
  imwrite(color_fpath, left_frame_cv_col);

  delete depth_cv;
}

// TODO(andrei): Refactor such that there is less code duplication between
// this method and `BuildKintinousLog`.
/// \brief Generates an InfiniTAM-readable color+depth image folder.
///
/// \param kitti_sequence_root Location of the KITTI or KITTI-odometry
/// video sequence.
/// \param frame_count The number of frames to process. Set to '-1' to
/// process all the frames in a folder.
/// \param use_color Whether to use the left color image as the image part
/// of the output. If set to 'false', the grayscale left image used to
/// compute the depth is used.
void BuildInfinitamLog(
    const fs::path &kitti_sequence_root,
    const fs::path &output_path,
    const int frame_count,
    const bool use_color,
    const double focal_length_px,
    const double stereo_baseline_m,
    const double min_depth_m,
    const double max_depth_m
) {
  if (!fs::exists(kitti_sequence_root)) {
    cerr << "Could not find KITTI dataset root directory at: "
         << kitti_sequence_root << "." << endl;
    return;
  }

  vector<StereoFrameFpaths> stereo_pair_fpaths = GetKittiStereoPairPaths(kitti_sequence_root);

  fs::path frames_folder = output_path / "Frames";
  if (!fs::exists(frames_folder)) {
    cout << "Output directory did not exist. Creating: " << frames_folder << endl;
    fs::create_directories(frames_folder);
  }

  int32_t num_frames = static_cast<int32_t>(stereo_pair_fpaths.size());
  cout << "Found: " << num_frames << " frames to process." << endl;

  ctpl::thread_pool p(6);
  for (int i = 0; i < stereo_pair_fpaths.size(); ++i) {
    if (frame_count > -1 && i >= frame_count) {
      cout << "Stopping early after reaching fixed limit of " << frame_count << " frames." << endl;
      break;
    }

    // Push the tasks to a pool and don't use any future, since the function automatically
    // does its job and writes the right file on disk.
    auto stereo_pair = stereo_pair_fpaths[i];
    p.push([=](int) {
      ProcessInfinitamFrame(i,
                            output_path,
                            use_color,
                            stereo_baseline_m,
                            focal_length_px,
                            min_depth_m,
                            max_depth_m,
                            stereo_pair);
    });
  }

  // Wait for the queue to be consumed and all tasks to be completed.
  p.stop(true);
}

double ReadFocalLength(fs::path kitti_odo_calib) {
  ifstream in(kitti_odo_calib);
  string dummy;
  double focal_length_px_x;
  in >> dummy >> focal_length_px_x;

  return focal_length_px_x;
}

} // namespace kitti2klg

/// \brief Helper to print a stacktrace on SIGSEGV.
void sig_handler(int sig) {
  void *btrace[10];
  int size = backtrace(btrace, 10);

  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(btrace, size, STDERR_FILENO);
  exit(1);
}

int main(int argc, char **argv) {
  namespace fs = std::experimental::filesystem;
  signal(SIGSEGV, sig_handler);

  // Processes the command line arguments, and populates the FLAGS_* variables
  // accordingly.
  gflags::SetUsageMessage("Stereo-to-RGBD conversion utility.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (kitti2klg::FLAGS_kitti_root.empty()) {
    std::cerr << "Please specify a KITTI root folder (--kitti_root=<folder>)." << std::endl;
    exit(1);
  }

  if (kitti2klg::FLAGS_baseline_meters < 0) {
    std::cerr << "Please specify the baseline (--baseline_meters=<double>)." << std::endl;
    exit(2);
  }

  if (kitti2klg::FLAGS_calib_file.empty()) {
    std::cerr << "Please specify a calibration file for creating the depth maps." << std::endl;
    exit(3);
  }

  fs::path kitti_seq_path = kitti2klg::FLAGS_kitti_root;
  fs::path output_path = kitti2klg::FLAGS_output;
  fs::path relative_calib_path = kitti2klg::FLAGS_calib_file;
  double focal_length_px = kitti2klg::ReadFocalLength(kitti_seq_path / relative_calib_path);

  std::cout << "Using a baseline of " << kitti2klg::FLAGS_baseline_meters << "m, a focal length of "
            << focal_length_px << " pixels, and keeping depth values in the range ["
            << kitti2klg::FLAGS_min_depth_meters << "--" << kitti2klg::FLAGS_max_depth_meters
            << "]." << std::endl;
  std::cout << "Loading KITTI pairs from folder [" << kitti_seq_path
            << "] and outputting ";

  if (kitti2klg::FLAGS_infinitam) {
    // Process the stereo data into an InfiniTAM-style folder consisting of
    // `pgm` RGB images and `pbm` grayscale depth images.
    // affine 0.0008 0.0 should be a good starting point in the InfiniTAM
    // calibration file (last line---stereo rig parameters).
    std::cout << "InfiniTAM-friendly pgm+pbm dir here: [" << output_path << "]" << std::endl;

    // TODO(andrei): Rename this whole program accordingly. We're no longer
    // focused on Kintinuous.
    kitti2klg::BuildInfinitamLog(kitti_seq_path,
                                 output_path,
                                 kitti2klg::FLAGS_process_frames,
                                 kitti2klg::FLAGS_use_color,
                                 kitti2klg::FLAGS_baseline_meters,
                                 focal_length_px,
                                 kitti2klg::FLAGS_min_depth_meters,
                                 kitti2klg::FLAGS_max_depth_meters);
  } else {
    // Process the stereo data into a Kintinuous-style binary logfile
    // consisting of JPEG-compressed RGB frames and zipped (!) depth channels
    std::cout << "Kintinuous-friendly *.klg file here: [" << output_path << "]." << std::endl;
    std::cerr << "NOT enabled at the moment, sorry." << std::endl;
//    kitti2klg::BuildKintinuousLog(kitti_seq_path, output_path, kitti2klg::FLAGS_process_frames);
  }

  return 0;
}
