
#include <experimental/filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

// TODO(andrei): If you decide to stick with gflags, mention them as a
// dependency in the README.
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <zlib.h>
#include <execinfo.h>
#include <csignal>
#include <limits>
#include <iomanip>
#include <wordexp.h>

#include "config.h"
#include "image.h"
#include "elas.h"
#include "util.h"

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

  /**
   * @brief Get a list of filenames for KITTI stereo pairs.
   * @param sequence_root The folder containing the desired KITTI sequence.
   * @param image_extension The extension of the image type to look for.
   *
   * @return A list of filename pairs containing the corresponding left and
   *    right grayscale image files for every frame.
   */
  vector<stereo_fpath_pair> GetKittiStereoPairPaths(
      const fs::path &sequence_root,
      const string image_extension = ".png"
  ) {
    fs::path left_dir = sequence_root / KITTI_GRAYSCALE_LEFT_FOLDER / "data";
    fs::path right_dir = sequence_root / KITTI_GRAYSCALE_RIGHT_FOLDER / "data";

    // Iterate through the left-image and the right-image directories
    // simultaneously, grabbing the file names along the way.
    vector<pair<fs::path, fs::path>> result;
    auto left_dir_it = fs::directory_iterator(left_dir);
    auto right_dir_it = fs::directory_iterator(right_dir);
    auto left_it = fs::begin(left_dir_it), right_it = fs::begin(right_dir_it);
    for(; left_it != fs::end(left_dir_it) && right_it != fs::end(right_dir_it);
        ++left_it, ++right_it) {
      if (kitti2klg::EndsWith(left_it->path().string(), image_extension) &&
          kitti2klg::EndsWith(right_it->path().string(), image_extension)) {
        result.emplace_back(left_it->path(), right_it->path());
      }
    }

    if (left_it != fs::end(left_dir_it) || right_it != fs::end(right_dir_it)) {
      throw runtime_error("Different frame counts in the two stereo folders.");
    }

    // Explicitly sort the paths so that they're in ascending order, since the
    // directory iterator does not guarantee it, and it's useful to iterate
    // through the frames in order when working on subsets of the data.
    auto compare_path_pairs = [](pair<fs::path, fs::path> stereo_pair_A,
                                 pair<fs::path, fs::path> stereo_pair_B) {
      return stereo_pair_A.first.filename().string() < stereo_pair_B.first.filename().string();
    };
    sort(result.begin(), result.end(), compare_path_pairs);

    return result;
  }

  /// Loads an image. Currently only PGM and PNG are supported. The image is
  /// loaded as grayscale, since that's all libelas cares about.
  ///
  /// \returns The loaded uchar image.
  img_ptr LoadImage(const fs::path &image_fpath) {
    if (! image_fpath.has_extension()) {
      throw runtime_error(Format(
          "Cannot load images without extensions, as format sniffing is "
          "currently not supported. Problematic path: %s", image_fpath.string()));
    }

    if (image_fpath.extension() == ".png") {
      cv::Mat img = cv::imread(image_fpath, CV_LOAD_IMAGE_ANYDEPTH);
      return make_shared<image<uchar>>(img.cols, img.rows, img.data);
    }
    else if (image_fpath.extension() == ".pgm") {
      return shared_ptr<image<uchar>>(loadPGM(image_fpath.string().c_str()));
    }
    else {
      throw runtime_error(Format(
          "Unsupported image format: %s", image_fpath.extension()));
    }
  }

  unique_ptr<stereo_image_pair> LoadStereoPair(const stereo_fpath_pair& pair_fpaths) {
    auto result = make_unique<stereo_image_pair>(
      LoadImage(pair_fpaths.first), LoadImage(pair_fpaths.second)
    );
    auto left = result->first;
    auto right = result->second;

    if (left->width()<=0 || left->height() <=0 || right->width()<=0 ||
        right->height() <= 0 || left->width()!=right->width() ||
        left->height()!=right->height()) {
      stringstream err;
      err << "ERROR: Images must be of same size, but" << endl
          << "       left: " << left->width() <<  " x " << left->height()
          << ", right: " << right->width() <<  " x " << right->height() << endl;

      // This also destroys the image pair unique_ptr.
      throw runtime_error(err.str());
    }

    return result;
  };

  /// \brief Computes a metric depth value from the supplied disparity and
  /// calibration parameters.
  uint16_t DepthFromDisparity(float disparity_px, float baseline_m, float focal_length_px) {
    if (disparity_px == kInvalidDepth) {
      // If libelas flags this as an invalid depth measurement, we want to
      // propagate that to the underlying SLAM system.
      return USHRT_MAX;
    }
    else {
      // Compute depth from disparity. Note that the scale is likely WAY
      // off, since the kinect sensor canonically spits things out in
      // millimiters, but God knows what InfiniTAM expects.
      // Z = (b * f) / disparity.

      // TODO(andrei): Double-check the fact that the disparity is expressed in pixels.
      // meters * px / px * 100 => centimeters.
      float depth_cm_f = ((baseline_m * focal_length_px) / disparity_px) *
          100.0f * 5.0f;
      uint16_t depth_cm_u = static_cast<uint16_t>(depth_cm_f);

      // Ensure we don't flag things really far away as invalid (==USHRT_MAX).
      if (depth_cm_u > USHRT_MAX - 1) {
//        if (depth_cm - 65535 > 500) {
//          badover++;
//        }
        depth_cm_u = USHRT_MAX - 1;
      }

      return depth_cm_u;
    }
  }


  /// \brief Uses libelas to compute the depth map from a stereo pair.
  /// Only computes the depth in the left camera's frame.
  /// \param depth_out The preallocated depth map object, to be populated by
  /// this function.
  void ComputeDepth(
      const image<uchar>& left,
      const image<uchar>& right,
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
    Elas::parameters param;
    param.postprocess_only_left = true;
    Elas elas(param);
    elas.process(left.data, right.data, D1_data, D2_data, dims);

    // TODO(andrei): Pass these as parameters, you MONSTER.
    float baseline_m = 0.571f;
    float focal_length_px = 645.2f;

    // Convert the float disparity map to 16-bit unsigned int depth map,
    // expressed in centimeters.
    for (int32_t i = 0; i < width * height; i++) {
      // Output 16-bit depth values, in the Kinect style.
      depth_out->data[i] = DepthFromDisparity(D1_data[i], baseline_m, focal_length_px);
    }

    free(D1_data);
    free(D2_data);
  }


  /// \brief Produces a Kintinuous-specific '.klg' file from a KITTI sequence.
  ///
  /// Reads through all stereo pairs of a dataset in order, computes the
  /// depth (in the left camera frame), and then takes that depth map, plus
  /// the left camera's image, as well as some other miscellaneous
  /// information, and writes it to a logfile which can be read by SLAM
  /// systems designed for RGBD input, such as Kintinuous.
  ///
  /// \param kitti_sequence_root Root folder of a particular sequence from
  /// the KITTI dataset.
  /// \param output_path '*.klg' file to write the log to. Overwrites
  /// existing files.
  /// \param process_frames Number of frames to process. -1 means all.
  ///
  /// The expected format first contains the number of frames, and then, for
  /// each frame in the sequence:
  ///  * int64_t: timestamp
  ///  * int32_t: depthSize
  ///  * int32_t: imageSize
  ///  * depthSize * unsigned char: depth_compress_buf
  ///  * imageSize * unsigned char: encodedImage->data.ptr
  ///
  /// \note Requires images from KITTI sequence to be in PGM format.
//  void BuildKintinuousLog(
//      const fs::path &kitti_sequence_root,
//      const fs::path &output_path,
//      const int process_frames) {
//    vector<pair<fs::path, fs::path>> stereo_pair_fpaths = GetKittiStereoPairPaths(kitti_sequence_root);
//    vector<long> timestamps = GetSequenceTimestamps(kitti_sequence_root);
//
//    // TODO(andrei): Flag to force resizing of all frames to arbitrary resolution.
//    // TODO(andrei): Split this method up into multiple chunks.
//    // TODO(andrei): If resizing is enabled, try computing the depth AFTER
//    // the resize. It may be slightly less accurate, but it could be much faster.
//
//    // Open the file and write the number of frames in the sequence.
//    FILE *log_file = fopen(output_path.string().c_str(), "wb+");
//    int32_t num_frames = static_cast<int32_t>(stereo_pair_fpaths.size());
//    if (process_frames > -1 && process_frames < num_frames) {
//      num_frames = process_frames;
//    }
//    fwrite(&num_frames, sizeof(int32_t), 1, log_file);
//
//    // KITTI dataset standard stereo image size: 1242 x 375.
//    int32_t standard_width = 1242;
//    int32_t standard_height = 375;
//
//    // The target size we should reshape our frames to be.
//    cv::Size target_size(640, 480);
//    size_t compressed_depth_buffer_size = standard_width * standard_height * sizeof(int16_t) * 4;
//    uint8_t *compressed_depth_buf = (uint8_t*) malloc(compressed_depth_buffer_size);
//
//    for(int i = 0; i < stereo_pair_fpaths.size(); ++i) {
//      if(process_frames > -1 && i >= process_frames) {
//        cout << "Stopping early after reaching fixed limit of " << process_frames
//             << " frames." << endl;
//        break;
//      }
//
//      const auto &pair_fpaths = stereo_pair_fpaths[i];
//      // Note: this is the timestamp associated with the left grayscale frame.
//      // The right camera frames have slightly different timestamps. For the
//      // purpose of this experimental application, we should nevertheless be OK
//      // to just use the left camera's timestamps.
//      cout << "Processing " << pair_fpaths.first << ", " << pair_fpaths.second;
//      auto img_pair = LoadStereoPair(pair_fpaths);
//      auto depth = make_shared<image<uchar>>(standard_width, standard_height);
//
//      // get image width and height
//      int32_t width  = img_pair->first->width();
//      int32_t height = img_pair->second->height();
//      if(width != standard_width || height != standard_height) {
//        throw runtime_error("Unexpected image dimensions encountered!");
//      }
//
//      int64_t frame_timestamp = timestamps[i];
//      ComputeDepth(img_pair->first, img_pair->second, depth.get());
//      // This is the value we give to zlib, which then updates it to reflect
//      // the resulting size of the data, after compression.
//      size_t compressed_depth_actual_size = compressed_depth_buffer_size;
//
//      cv::Mat depth_cv = cv::Mat(cvSize(depth->width(), depth->height()),
//                                CV_8U,
//                                depth->data);
//
////      // Invert the depth mask, since Kintinuous uses a different convention.
////      cv::Mat zero_mask = (depth_cv == 0);
////      cv::subtract(cv::Scalar::all(255.0), depth_cv, depth_cv);
////      // We must ensure that invalid pixels are still set to zero.
////      depth_cv.setTo(cv::Scalar::all(0.0), zero_mask);
//
//      // TODO(andrei): Consider moving these depth map operations to the depth
//      // map generation function.
//      // Try to mark measurements which are too far away as invalid, since
//      // otherwise they can corrupt Kintinuous, it seems.
//      // TODO(andrei): Investigate this further.
//      cv::Mat far_mask = (depth_cv > 200);
//      depth_cv.setTo(cv::Scalar::all(0.0), far_mask);
//
//      cv::Mat depth_cv_16_bit(depth_cv.size(), CV_16U);
//      cv::Mat depth_cv_vga(target_size, CV_16U);
//
//      // This parameter ensures the full depth range of 0-255 is transferred
//      // properly when we switch to 16-bit depth (required by Kintinuous).
//      double alpha = 255.0;
//
//      // This is a parameter controlling the range of our depth. There should
//      // be ways of setting this based on, e.g., our stereo rig configuration.
//      double scale = 0.15;
//
//      // VERY IMPORTANT: You get very funky results in Kintinuous if you
//      // accidentally give it an 8-bit depth map. It misinterprets it by sort
//      // of splitting it up into what looks like footage meant for VR, i.e.,
//      // into two depthmaps. Make sure you give Kintinuous 16-bit depth!
//      // Ensure that our depth map is 16-bit, NOT 8-bit.
//      depth_cv.convertTo(depth_cv_16_bit, CV_16U, alpha * scale);
//      cv::resize(depth_cv_16_bit, depth_cv_vga, target_size);
//      size_t raw_depth_size = depth_cv_vga.total() * depth_cv_vga.elemSize();
//
//      // Warning: 'compressed_depth_buf' will contain junk and residue from
//      // previous frames beyond the indicated 'compressed_depth_actual_size'!
//      // TODO(andrei): Try NO compression (NO JPEG and NO zlib). Kintinuous
//      // does support that, and it may not be necessary.
//      check_compress2(compress2(
//          compressed_depth_buf,
//          &compressed_depth_actual_size,
//          (const Bytef*) depth_cv_vga.data,
//          raw_depth_size,
//          Z_BEST_SPEED));
//
////      float compression_ratio =
////          static_cast<float>(compressed_depth_actual_size) / raw_depth_size;
////      cout << "Depth compressed OK. Compressed result size: "
////           << compressed_depth_actual_size << "/" << raw_depth_size
////           << " (Compression: " << compression_ratio * 100.0 << "%)" << endl;
//
//      // Encode the left frame as a JPEG for the log.
//      cv::Mat left_frame_cv = cv::Mat(cvSize(img_pair->first->width(),
//                                             img_pair->first->height()),
//                                      CV_8U,
//                                      img_pair->first->data);
//
//      // TODO(andrei): Kintinuous is reading CV_8UC1. Should all our images use that format?
//      cv::Mat left_frame_vga;
//      cv::resize(left_frame_cv, left_frame_vga, target_size);
//      CvMat *encoded_rgb_jpeg_vga = EncodeJpeg(left_frame_vga);
//
//      // The encoded JPEG is stored as a row-matrix.
//      int32_t jpeg_size = static_cast<int32_t>(encoded_rgb_jpeg_vga->width);
//
//      // Write all the current frame information to the logfile.
//      fwrite(&frame_timestamp, sizeof(int64_t), 1, log_file);
//      fwrite(&compressed_depth_actual_size, sizeof(int32_t), 1, log_file);
//      fwrite(&jpeg_size, sizeof(int32_t), 1, log_file);
//      fwrite(compressed_depth_buf, compressed_depth_actual_size, 1, log_file);
//      fwrite(encoded_rgb_jpeg_vga->data.ptr, static_cast<size_t>(jpeg_size), 1, log_file);
//
//      cout << " Write OK." << endl;
//      cvReleaseMat(&encoded_rgb_jpeg_vga);
//    }
//
//    free(compressed_depth_buf);
//    fflush(log_file);
//    fclose(log_file);
//  }

  // TODO(andrei): Refactor such that there is less code duplication between
  // this method and `BuildKintinousLog`.
  void BuildInfinitamLog(
      const fs::path &kitti_sequence_root,
      const fs::path &output_path,
      const int process_frames
  ) {
    if (! fs::exists(kitti_sequence_root)) {
      cerr << "Could not find KITTI dataset root directory at: "
           << kitti_sequence_root << "." << endl;
      return;
    }

    vector<stereo_fpath_pair> stereo_pair_fpaths = GetKittiStereoPairPaths(kitti_sequence_root);

    fs::path frames_folder = output_path / "Frames";
    if (! fs::exists(frames_folder)) {
      cout << "Output directory did not exist. Creating: " << frames_folder
           << endl;
      fs::create_directories(frames_folder);
    }

    int32_t num_frames = static_cast<int32_t>(stereo_pair_fpaths.size());
    cout << "Found: " << num_frames << " frames to process." << endl;


    // The target size we should reshape our frames to be.
    cv::Size target_size(640, 480);

    for (int i = 0; i < stereo_pair_fpaths.size(); ++i) {
      if (process_frames > -1 && i >= process_frames) {
        cout << "Stopping early after reaching fixed limit of "
             << process_frames
             << " frames." << endl;
        break;
      }

      const auto &pair_fpaths = stereo_pair_fpaths[i];
      // Note: this is the timestamp associated with the left grayscale frame.
      // The right camera frames have slightly different timestamps. For the
      // purpose of this experimental application, we should nevertheless be OK
      // to just use the left camera's timestamps.
      cout << "Processing " << pair_fpaths.first << ", " << pair_fpaths.second;
      auto img_pair = LoadStereoPair(pair_fpaths);

      // get image width and height
      int32_t width  = img_pair->first->width();
      int32_t height = img_pair->second->height();
      if(width != kKittiFrameWidth || height != kKittiFrameHeight) {
        throw runtime_error(Format(
            "Unexpected image dimensions encountered! Was assuming standard "
            "KITTI frame dimensions of %d x %d.", kKittiFrameWidth, kKittiFrameHeight));
      }

      auto depth = make_shared<image<uint16_t>>(width, height);
      ComputeDepth(*(img_pair->first), *(img_pair->second), depth.get());
      cv::Mat *depth_cv = ToCvMat(*depth);

      cv::Mat depth_cv_resized(target_size, CV_16U);
      cv::resize(*depth_cv, depth_cv_resized, target_size);

      ostringstream grayscale_fname;
      grayscale_fname << setfill('0') << setw(4) << i << ".pgm";
      stringstream color_fname;
      color_fname << setfill('0') << setw(4) << i << ".ppm";

      fs::path grayscale_fpath = output_path / "Frames" / grayscale_fname.str();
      fs::path color_fpath = output_path / "Frames" / color_fname.str();

      cv::Mat *left_frame_cv = ToCvMat(*(img_pair->first));
      cv::Mat left_frame_cv_col(left_frame_cv->size(), CV_8U);
      cv::cvtColor(*left_frame_cv, left_frame_cv_col, CV_GRAY2BGR);

      cv::Mat left_frame_vga;
      cv::resize(left_frame_cv_col, left_frame_vga, target_size);

      cv::imwrite(color_fpath, left_frame_vga);
      cv::imwrite(grayscale_fpath, depth_cv_resized);

      delete left_frame_cv;
      delete depth_cv;
      cout << " OK." << endl;
    }
  }
}

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

  if(kitti2klg::FLAGS_kitti_root.empty()) {
    std::cerr << "Please specify a KITTI root folder (--kitti_root=<folder>)."
              << std::endl;
    exit(1);
  }

  fs::path kitti_seq_path = kitti2klg::FLAGS_kitti_root;
  fs::path output_path = kitti2klg::FLAGS_output;

  std::cout << "Loading KITTI pairs from folder [" << kitti_seq_path
            << "] and outputting ";

  if (kitti2klg::FLAGS_infinitam) {
    // Process the stereo data into an InfiniTAM-style folder consisting of
    // `pgm` RGB images and `pbm` grayscale depth images.
    std::cout << "InfiniTAM-friendly pgm+pbm dir here: [" << output_path << "]" << std::endl;

    // TODO(andrei): Rename this package accordingly.
    kitti2klg::BuildInfinitamLog(kitti_seq_path, output_path, kitti2klg::FLAGS_process_frames);
  }
  else {
    // Process the stereo data into a Kintinuous-style binary logfile
    // consisting of JPEG-compressed RGB frames and zipped (!) depth channels
    std::cout << "Kintinuous-friendly *.klg file here: [" << output_path << "]." << std::endl;
    std::cerr << "NOT enabled at the moment, sorry." << std::endl;
//    kitti2klg::BuildKintinuousLog(kitti_seq_path, output_path, kitti2klg::FLAGS_process_frames);
  }

  return 0;
}
