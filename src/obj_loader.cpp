#include "obj_loader.hpp"

OBJModel OBJLoader::load(const std::string& path) {
    OBJModel model;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << path << std::endl;
        return model;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            model.vertices.emplace_back(x, y, z);
        } else if (prefix == "f") {
            for (int i = 0; i < 3; i++) {
                std::string token;
                iss >> token;
                std::istringstream tokStream(token);
                unsigned int vIndex;
                tokStream >> vIndex;
                model.indices.push_back(vIndex - 1); // OBJ is 1-indexed
            }
        }
    }

    std::cout << "Loaded OBJ: " << model.vertices.size()
              << " vertices, " << model.indices.size() / 3
              << " triangles." << std::endl;

    return model;
}
