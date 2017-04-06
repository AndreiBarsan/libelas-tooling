#include <wordexp.h>

#include <experimental/filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <zlib.h>

#include "image.h"
#include "elas.h"
#include "util.h"

namespace kitti2klg {
  using namespace std;
  namespace fs = std::experimental::filesystem;
  using stereo_fpath_pair = std::pair<fs::path, fs::path>;
  using stereo_image_pair = std::pair<image<uchar>*, image<uchar>*>;

  const string KITTI_GRAYSCALE_LEFT_FOLDER  = "image_00";
  const string KITTI_GRAYSCALE_RIGHT_FOLDER = "image_01";
  const string KITTI_COLOR_LEFT_FOLDER      = "image_02";
  const string KITTI_COLOR_RIGHT_FOLDER     = "image_03";

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
      const string image_extension = ".pgm"
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
    // directory iterator does not guarantee it.
    auto compare_path_pairs = [](pair<fs::path, fs::path> stereo_pair_A,
                                 pair<fs::path, fs::path> stereo_pair_B) {
      return stereo_pair_A.first.filename().string() < stereo_pair_B.first.filename().string();
    };
    sort(result.begin(), result.end(), compare_path_pairs);

    return result;
  };;

  /**
   * NOTE: This only looks at the timestamps associated with the left
   * greyscale images. There are also timestamps associated with each of the
   * other cameras, and they are not exactly identitcal. Nevertheless, this
   * approach should be good enough for the current application.
   *
   * @return A vector of UTC timestamps at MICROSECOND resolution, necessary
   *    for the custom Kintinuous `.klg` format.
   */
  vector<long> GetSequenceTimestamps(const fs::path &root) {
    fs::path timestamp_fpath = root;
    timestamp_fpath.append(KITTI_GRAYSCALE_LEFT_FOLDER);
    timestamp_fpath.append("timestamps.txt");

    ifstream in(timestamp_fpath);

    vector<long> timestamps;
    string chunk;
    while (getline(in, chunk, '\n')) {
      tm time;
      long nanosecond;
      ReadTimestampWithNanoseconds(chunk, &time, &nanosecond);

      // The format expected by Kintinuous uses microsecond resolution.
      long microsecond = nanosecond / 1000;
      time_t total_seconds = timegm(&time);
      long total_microseconds = total_seconds * 1000 * 1000 + microsecond;

      timestamps.push_back(total_microseconds);
    }

    return timestamps;
  };

  /// Loads an image. Currently only PGM is supported.
  image<uchar>* LoadImage(const fs::path &image_fpath) {
    // TODO(andrei): Use OpenCV, if necessary, and convert to image<uchar>.
    // Since the 'loadPGM' code just reads the raw PGM file, extracting width
    // and height and then and putting it into an array of uchars, it shouldn't
    // be very difficult.
    return loadPGM(image_fpath.string().c_str());
  }

  unique_ptr<pair<image<uchar>*, image<uchar>*>> LoadStereoPair(
      const pair<fs::path, fs::path>& pair_fpaths
  ) {
    auto result = make_unique<pair<image<uchar>*, image<uchar>*>>(
      LoadImage(pair_fpaths.first), LoadImage(pair_fpaths.second)
    );
    auto left = result->first, right = result->second;

    if (left->width()<=0 || left->height() <=0 || right->width()<=0 || right->height() <=0 ||
        left->width()!=right->width() || left->height()!=right->height()) {
      stringstream err;
      err << "ERROR: Images must be of same size, but" << endl
          << "       left: " << left->width() <<  " x " << left->height()
          << ", right: " << right->width() <<  " x " << right->height() << endl;

      // This also destroys the unique_ptr.
      throw runtime_error(err.str());
    }

    return result;
  };

  /// \brief Uses libelas to compute the depth map from a stereo pair.
  /// Only computes the depth in the left camera's frame.
  /// \param depth_out The preallocated depth map object, to be populated by
  /// this function.
  void ComputeDepth(
      const image<uchar>* left,
      const image<uchar>* right,
      image<uchar> *depth_out
  ) {
    // Heavily based on the demo program which ships with libelas.
    int32_t width = left->width();
    int32_t height = left->height();

    // allocate memory for disparity image
    const int32_t dims[3] = {width, height, width}; // bytes per line = width
    float *D1_data = (float *) malloc(width * height * sizeof(float));
    // 'D2_data' is necessary inside libelas, but not used by our code.
    float *D2_data = (float *) malloc(width * height * sizeof(float));

    // process
    Elas::parameters param;
    param.postprocess_only_left = true;
    Elas elas(param);
    elas.process(left->data, right->data, D1_data, D2_data, dims);

    // Find maximum disparity for scaling output disparity images to [0..255].
    float disp_max = 0;
    for (int32_t i = 0; i < width * height; i++) {
      if (D1_data[i] > disp_max) disp_max = D1_data[i];
    }

    // Copy float to uchar, after applying the [0..255] scaling.
    for (int32_t i = 0; i < width * height; i++) {
      depth_out->data[i] = (uint8_t) max(255.0 * D1_data[i] / disp_max, 0.0);
    }

    free(D1_data);
    free(D2_data);
  }

  /// \brief Encodes a raw image as JPEG, for use in the 'klg' log.
  CvMat* EncodeJpeg(const image<uchar> *const raw_image) {
    int jpeg_params[] = {CV_IMWRITE_JPEG_QUALITY, 90, 0};
    int width = raw_image->width();
    int height = raw_image->height();

//    cv::Vec<unsigned char, 1> *raw_gray_data = (cv::Vec<unsigned char, 1> *) raw_image->data;
//    cv::Mat1b left_frame_mat(height, width, raw_gray_data->val);
//    imencode(".jpg", left_frame_mat, out_jpeg_buf, jpeg_params);

    // Small hack to ensure our dump has RGB channels, even if here, in this
    // tool, we're just dealing with grayscale.
    // TODO(andrei): Use grayscale pairs for depth as before, but pass the
    // actual color left frame to this function.
    IplImage *color_ipl_img = cvCreateImage(
        cvSize(raw_image->width(), raw_image->height()), 
        IPL_DEPTH_8U,
        3);
    IplImage *ipl_img = new IplImage();
    cvCvtColor(ipl_img, color_ipl_img, CV_GRAY2BGR);


    // We use the C-style API to be consistend with the Kintinuous ecosystem.
    cv::Mat1b gray(height, width, raw_image->data);
    IplImage *img = new IplImage(gray);
    CvMat *encoded = cvEncodeImage(".jpg", img, jpeg_params);
    return encoded;
  }

  /// Checks the return result of zlib's `compress2` function, throwing an
  /// error of compression failed.
  void check_compress2(int compress_result) {
    if(compress_result == Z_BUF_ERROR) {
      throw runtime_error("zlib Z_BUF_ERROR: Destination buffer too small.");
    }
    else if(compress_result == Z_MEM_ERROR) {
      throw runtime_error("zlib Z_MEM_ERROR; Insufficient memory.");
    }
    else if(compress_result == Z_STREAM_ERROR) {
      throw runtime_error("zlib Z_STREAM_ERROR: Unknown compression level.");
    }
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
  void BuildKintinuousLog(const fs::path &kitti_sequence_root) {
    vector<pair<fs::path, fs::path>> stereo_pair_fpaths = GetKittiStereoPairPaths(kitti_sequence_root);
    vector<long> timestamps = GetSequenceTimestamps(kitti_sequence_root);

    fs::path fpath = "test_dump.klg";
    FILE *log_file = fopen(fpath.string().c_str(), "wb+");
    int32_t num_frames = static_cast<int32_t>(stereo_pair_fpaths.size());
    fwrite(&num_frames, sizeof(int32_t), 1, log_file);

    // KITTI dataset standard stereo image size: 1242 x 375.
    int32_t standard_width = 1242;
    int32_t standard_height = 375;
    size_t compressed_depth_buffer_size = standard_width * standard_height * sizeof(int16_t) * 4;
    uint8_t *compressed_depth_buf = (uint8_t*) malloc(compressed_depth_buffer_size);

    for(int i = 0; i < stereo_pair_fpaths.size(); ++i) {
      const auto &pair_fpaths = stereo_pair_fpaths[i];
      // Note: this is the timestamp associated with the left grayscale frame.
      // The right camera frames have slightly different timestamps. For the
      // purpose of this experimental application, we should nevertheless be OK
      // to just use the left camera's timestamps.
      cout << "Processing " << pair_fpaths.first << ", " << pair_fpaths.second << flush;
      auto img_pair = LoadStereoPair(pair_fpaths);
      auto depth = make_shared<image<uchar>>(standard_width, standard_height);

      // get image width and height
      int32_t width  = img_pair->first->width();
      int32_t height = img_pair->second->height();
      if(width != standard_width || height != standard_height) {
        throw runtime_error("Unexpected image dimensions encountered!");
      }

      int64_t frame_timestamp = timestamps[i];
      size_t raw_depth_size = width * height * sizeof(short);
      ComputeDepth(img_pair->first, img_pair->second, depth.get());
      // This is the value we give to zlib, which then updates it to reflect
      // the resulting size of the data, after compression.
      size_t compressed_depth_actual_size = compressed_depth_buffer_size;

      // Warning: 'compressed_depth_buf' will contain junk and residue from
      // previous frames beyond the indicated 'compressed_depth_actual_size'!
      check_compress2(compress2(
          compressed_depth_buf,
          &compressed_depth_actual_size,
          (const Bytef*) depth->data,
          raw_depth_size,
          Z_BEST_SPEED));

//      float compression_ratio =
//          static_cast<float>(compressed_depth_actual_size) / raw_depth_size;
//      cout << "Depth compressed OK. Compressed result size: "
//           << compressed_depth_actual_size << "/" << raw_depth_size
//           << " (Compression: " << compression_ratio * 100.0 << "%)" << endl;

      // Encode the left frame as a JPEG for the log.
      vector<uchar> out_jpeg_buf;
      EncodeJpeg(img_pair->first, out_jpeg_buf);
      int32_t jpeg_size = static_cast<int32_t>(out_jpeg_buf.size());

      cout << " OK." << flush;

      // Write all the current frame information to the logfile.
      fwrite(&frame_timestamp, sizeof(int64_t), 1, log_file);
      fwrite(&compressed_depth_actual_size, sizeof(int32_t), 1, log_file);
      fwrite(&jpeg_size, sizeof(int32_t), 1, log_file);
      fwrite(compressed_depth_buf, compressed_depth_actual_size, 1, log_file);
      fwrite(out_jpeg_buf.data(), out_jpeg_buf.size(), 1, log_file);

      cout << " Write OK." << endl;
    }

    free(compressed_depth_buf);
    fflush(log_file);
    fclose(log_file);
  }


}

int main() {
  namespace fs = std::experimental::filesystem;

  std::cout << "Loading KITTI pairs from folder [] and outputting "
      "ElasticFusion/Kintinuous-friendly *.klg file." << std::endl;

  fs::path kitti_root = kitti2klg::GetExpandedPath("~/datasets/kitti");
  fs::path kitti_seq_path = kitti_root / "2011_09_26" / "2011_09_26_drive_0095_sync";
  kitti2klg::BuildKintinuousLog(kitti_seq_path);

  return 0;
}
