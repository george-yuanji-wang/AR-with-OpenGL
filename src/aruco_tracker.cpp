// aruco_tracker.cpp
#include "aruco_tracker.hpp"

ArucoTracker::ArucoTracker(float markerLengthMeters, const cv::Mat& K, const cv::Mat& dist)
: markerLen_(markerLengthMeters) {
    K.convertTo(K_, CV_64F);
    dist.convertTo(dist_, CV_64F);
}

bool ArucoTracker::detect(const cv::Mat& frameBGR, cv::Vec3d& rvec, cv::Vec3d& tvec) {
    cv::Mat gray;
    cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);

    static cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    static cv::aruco::DetectorParameters detectorParams;
    static cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    std::vector<std::vector<cv::Point2f>> corners, rejected;
    std::vector<int> ids;
    detector.detectMarkers(gray, corners, ids, rejected);
    if (ids.empty()) return false;

    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.at<cv::Vec3f>(0,0) = cv::Vec3f(-markerLen_/2.f,  markerLen_/2.f, 0);
    objPoints.at<cv::Vec3f>(1,0) = cv::Vec3f( markerLen_/2.f,  markerLen_/2.f, 0);
    objPoints.at<cv::Vec3f>(2,0) = cv::Vec3f( markerLen_/2.f, -markerLen_/2.f, 0);
    objPoints.at<cv::Vec3f>(3,0) = cv::Vec3f(-markerLen_/2.f, -markerLen_/2.f, 0);

    std::vector<cv::Vec3d> rvecs(corners.size()), tvecs(corners.size());
    for (size_t i = 0; i < corners.size(); ++i) {
        cv::solvePnP(objPoints, corners[i], K_, dist_, rvecs[i], tvecs[i], false, cv::SOLVEPNP_ITERATIVE);
    }

    rvec = rvecs[0];
    tvec = tvecs[0];
    return true;
}
