#include "aruco_tracker.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

static bool loadCalibration(cv::Mat& K, cv::Mat& dist) {
    cv::FileStorage fs("camera.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> K;
    fs["dist_coeffs"] >> dist;
    return !K.empty();
}

int main(int argc, char** argv) {
    const float markerLength = 0.05f; // meters
    const int camId = 0;

    cv::VideoCapture inputVideo;
    int waitTime = 10;
    if (argc > 1) {
        inputVideo.open(argv[1]);
        waitTime = 0;
    } else {
        inputVideo.open(camId);
        waitTime = 10;
    }
    if (!inputVideo.isOpened()) { std::cerr << "Failed to open input\n"; return -1; }

    int w = (int)inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (w <= 0 || h <= 0) { w = 640; h = 480; }

    cv::Mat K, dist;
    if (!loadCalibration(K, dist)) {
        double fx = (w/2.0) / std::tan(60.0 * CV_PI / 360.0);
        double fy = fx;
        double cx = w/2.0, cy = h/2.0;
        K = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
        dist = cv::Mat::zeros(1,5,CV_64F);
        std::cout << "track_test: using approximate intrinsics (no camera.yml)\n";
    }

    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.at<cv::Vec3f>(0,0) = cv::Vec3f(-markerLength/2.f,  markerLength/2.f, 0);
    objPoints.at<cv::Vec3f>(1,0) = cv::Vec3f( markerLength/2.f,  markerLength/2.f, 0);
    objPoints.at<cv::Vec3f>(2,0) = cv::Vec3f( markerLength/2.f, -markerLength/2.f, 0);
    objPoints.at<cv::Vec3f>(3,0) = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::DetectorParameters detectorParams;
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    ArucoTracker tracker(markerLength, K, dist);

    double totalTime = 0.0;
    int totalIterations = 0;

    for (;;) {
        if (!inputVideo.grab()) break;
        cv::Mat image; inputVideo.retrieve(image);
        if (image.empty()) break;

        const double tick = (double)cv::getTickCount();

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        detector.detectMarkers(image, corners, ids, rejected);

        const size_t nMarkers = corners.size();
        std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);
        if (!ids.empty()) {
            for (size_t i = 0; i < nMarkers; ++i) {
                cv::solvePnP(objPoints, corners[i], K, dist, rvecs[i], tvecs[i], false, cv::SOLVEPNP_ITERATIVE);
            }
        }

        const double currentTime = ((double)cv::getTickCount() - tick) / cv::getTickFrequency();
        totalTime += currentTime;
        totalIterations++;

        cv::Mat imageCopy; image.copyTo(imageCopy);
        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
            for (size_t i = 0; i < ids.size(); ++i) {
                cv::drawFrameAxes(imageCopy, K, dist, rvecs[i], tvecs[i], markerLength * 1.5f, 2);
            }
        }
        if (!rejected.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, rejected, cv::noArray(), cv::Scalar(100,0,255));
        }

        cv::Vec3d r_test, t_test;
        bool ok = tracker.detect(image, r_test, t_test);
        if (ok) {
            cv::drawFrameAxes(imageCopy, K, dist, r_test, t_test, markerLength, 2);
            cv::putText(imageCopy, "ArucoTracker OK", {10,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        } else {
            cv::putText(imageCopy, "ArucoTracker NO DETECTION", {10,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
        }

        cv::imshow("track_test", imageCopy);
        char key = (char)cv::waitKey(waitTime);
        if (key == 27) break;
    }

    return 0;
}
