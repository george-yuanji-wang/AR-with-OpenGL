// aruco_tracker.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

class ArucoTracker {
public:
    ArucoTracker(float markerLengthMeters, const cv::Mat& K, const cv::Mat& dist);
    bool detect(const cv::Mat& frameBGR, cv::Vec3d& rvec, cv::Vec3d& tvec);

private:
    float markerLen_;
    cv::Mat K_, dist_;
};

