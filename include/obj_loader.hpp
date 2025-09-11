#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

struct OBJModel {
    std::vector<cv::Point3f> vertices;
    std::vector<unsigned int> indices;
};

class OBJLoader {
public:
    static OBJModel load(const std::string& path);
};
