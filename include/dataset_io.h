#ifndef DATASET_IO_H
#define DATASET_IO_H

#include <string>
#include <ros/ros.h>
#include <regex>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
// #include <array>
#include <utils.hpp>
#include "BALM/tools.hpp"
#include "BALM/bavoxel.hpp"

namespace lvba {


class DatasetIO {
public: 
    
    DatasetIO(ros::NodeHandle& nh);
    ~DatasetIO() = default;

    void readParameters(ros::NodeHandle &nh);
    bool loadDataset();
    bool handleImages();
    bool handleBodyPoints();
    bool handleLidarPoses();
    bool handleCamPoses();
    // bool handlePoints();
    // bool handlePoses();
    // bool handleVoxelId();
    bool loadPosesTUM(const std::string& file, size_t STRIDE, std::vector<Sophus::SE3>& poses_out);
    // std::string getImagePath(double image_id);
    // std::string getPcdPath(double pcd_id);

    void undistortImage(const cv::Mat& raw, cv::Mat& rectified);
    
    cv::Mat cvK_, cvD_;
    cv::Mat undist_map1_, undist_map2_;

    int width_, height_;
    double fx_, fy_, cx_, cy_;
    double k1_, k2_, p1_, p2_;
    double resize_scale_;

    std::vector<double> extrinT_;
    std::vector<double> extrinR_;
    std::vector<double> cameraextrinT_;
    std::vector<double> cameraextrinR_;

    std::vector<double> images_ids_;
    // pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_;
    // std::vector<Sophus::SE3> poses_;

    std::vector<Sophus::SE3> lidar_poses_;
    std::vector<Sophus::SE3> image_poses_;
    
    std::vector<std::vector<VOXEL_LOC>> all_voxel_ids_;
    std::string dataset_path_;
    std::string colmap_db_path_;
    int image_stride_;

    std::vector<pcl::PointCloud<PointType>::Ptr> pl_fulls_;
    std::vector<IMUST> x_buf_;
    std::vector<IMUST> x_buf_before_;

    bool window_ba_enable_ = true;
    int window_ba_size_ = 10;
    double anchor_leaf_size_ = 0.1;
    bool use_window_ba_rel_ = false;

    bool stage1_enable_ = true;
    double stage1_root_voxel_size_ = 0.5;
    std::vector<float> stage1_eigen_ratio_array_ = {0.3f, 0.1f, 0.06f, 0.03f};

    double stage2_root_voxel_size_ = 0.5;
    std::vector<float> stage2_eigen_ratio_array_ = {0.08f, 0.08f, 0.08f, 0.08f};

};
typedef std::shared_ptr<DatasetIO> DatasetIOPtr;

}
#endif // DATASET_IO_H
