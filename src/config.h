
#ifndef LIBELAS_CONFIG_H_H
#define LIBELAS_CONFIG_H_H

#include <string>

namespace kitti2klg {
  const std::string KITTI_GRAYSCALE_LEFT_FOLDER = "image_00";
  const std::string KITTI_GRAYSCALE_RIGHT_FOLDER = "image_01";
  const std::string KITTI_COLOR_LEFT_FOLDER = "image_02";
  const std::string KITTI_COLOR_RIGHT_FOLDER = "image_03";

  // KITTI dataset standard stereo image size: 1242 x 375.
  static const int kKittiFrameWidth = 1242;
  static const int kKittiFrameHeight = 375;
}


#endif //LIBELAS_CONFIG_H_H
