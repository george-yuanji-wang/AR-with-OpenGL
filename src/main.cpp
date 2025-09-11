#include "aruco_tracker.hpp"
#include "obj_loader.hpp"

#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdint>

static bool  showModel = false;            
static const float MARKER_LEN = 0.05f;     

static GLuint  vaoModel = 0;
static GLuint  vboModel = 0;
static GLuint  eboModel = 0;
static GLsizei idxCount = 0;
static bool    modelReady = false;

static GLuint makeShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){ char log[1024]; glGetShaderInfoLog(s,1024,nullptr,log); std::cerr<<log<<"\n"; }
    return s;
}
static GLuint makeProgram(const char* vs, const char* fs, const char* posName, const char* uvName=nullptr) {
    GLuint v = makeShader(GL_VERTEX_SHADER, vs);
    GLuint f = makeShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    if (posName) glBindAttribLocation(p, 0, posName);
    if (uvName)  glBindAttribLocation(p, 1, uvName);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){ char log[1024]; glGetProgramInfoLog(p,1024,nullptr,log); std::cerr<<log<<"\n"; }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static const char* VS_BG = R"(
#version 150
in vec2 aPos;
in vec2 aUV;
out vec2 vUV;
void main(){ vUV=aUV; gl_Position=vec4(aPos,0.0,1.0); }
)";
static const char* FS_BG = R"(
#version 150
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main(){ fragColor = texture(uTex, vUV); }
)";
static const char* VS_GEOM = R"(
#version 150
in vec3 aPos;
uniform mat4 uP;
uniform mat4 uMV;
void main(){ gl_Position = uP * uMV * vec4(aPos, 1.0); }
)";
static const char* FS_GEOM = R"(
#version 150
uniform vec3 uColor;
out vec4 fragColor;
void main(){ fragColor = vec4(uColor, 1.0); }
)";

static void toGLColumnMajor(const cv::Matx44f& M, float out16[16]) {
    out16[0]=M(0,0); out16[1]=M(1,0); out16[2]=M(2,0); out16[3]=M(3,0);
    out16[4]=M(0,1); out16[5]=M(1,1); out16[6]=M(2,1); out16[7]=M(3,1);
    out16[8]=M(0,2); out16[9]=M(1,2); out16[10]=M(2,2); out16[11]=M(3,2);
    out16[12]=M(0,3); out16[13]=M(1,3); out16[14]=M(2,3); out16[15]=M(3,3);
}
static cv::Matx44f mul44(const cv::Matx44f& A, const cv::Matx44f& B){
    cv::Matx44f C = cv::Matx44f::zeros();
    for(int r=0;r<4;r++) for(int c=0;c<4;c++){
        float s=0; for(int k=0;k<4;k++) s+=A(r,k)*B(k,c);
        C(r,c)=s;
    }
    return C;
}
static cv::Matx44f makeScale(float sx, float sy, float sz){
    cv::Matx44f S = cv::Matx44f::eye();
    S(0,0)=sx; S(1,1)=sy; S(2,2)=sz;
    return S;
}
static cv::Matx44f makeTranslate(float tx, float ty, float tz){
    cv::Matx44f T = cv::Matx44f::eye();
    T(0,3)=tx; T(1,3)=ty; T(2,3)=tz;
    return T;
}
static cv::Matx44f projFromK(const cv::Mat& K, int w, int h, float n, float f) {
    double fx=K.at<double>(0,0), fy=K.at<double>(1,1), cx=K.at<double>(0,2), cy=K.at<double>(1,2);
    cv::Matx44f P = cv::Matx44f::zeros();
    P(0,0) =  2.0f * float(fx) / float(w);
    P(1,1) =  2.0f * float(fy) / float(h);
    P(0,2) =  1.0f - 2.0f * float(cx) / float(w);
    P(1,2) =  2.0f * float(cy) / float(h) - 1.0f;
    P(2,2) = -(f + n) / (f - n);
    P(2,3) = -2.0f * f * n / (f - n);
    P(3,2) = -1.0f;
    return P;
}

static void fbsize(GLFWwindow* win, int fbw, int fbh) { glViewport(0, 0, fbw, fbh); }
static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_SPACE) showModel = !showModel;
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, 1);
    }
}

static bool loadCalibration(cv::Mat& K, cv::Mat& dist) {
    cv::FileStorage fs("camera.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> K;
    fs["dist_coeffs"] >> dist;
    return !K.empty();
}

int main() {
    cv::VideoCapture cap(0);
    if(!cap.isOpened()){ std::cerr<<"Camera open failed\n"; return -1; }
    int camW=640, camH=480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, camW);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,camH);

    cv::Mat K, dist;
    if (!loadCalibration(K, dist)) {
        double fx = (camW/2.0) / std::tan(60.0 * CV_PI / 360.0);
        double fy = fx;
        double cx = camW/2.0, cy = camH/2.0;
        K = (cv::Mat_<double>(3,3) << fx,0,cx,  0,fy,cy,  0,0,1);
        dist = cv::Mat::zeros(1,5,CV_64F);
        std::cout << "Using approximate intrinsics\n";
    }

    if(!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,2);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
    GLFWwindow* win = glfwCreateWindow(camW,camH,"AR base",nullptr,nullptr);
    if(!win){ glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetKeyCallback(win, keyCallback);
    glEnable(GL_DEPTH_TEST);
    int fbw=0, fbh=0; glfwGetFramebufferSize(win,&fbw,&fbh); glViewport(0,0,fbw,fbh);
    glfwSetFramebufferSizeCallback(win, fbsize);

    GLuint progBG = makeProgram(VS_BG, FS_BG, "aPos", "aUV");
    float quad[] = { -1.f,-1.f,0.f,1.f,  1.f,-1.f,1.f,1.f,  -1.f, 1.f,0.f,0.f,  1.f, 1.f,1.f,0.f };
    GLuint vaoBG=0,vboBG=0,tex=0;
    glGenVertexArrays(1,&vaoBG);
    glGenBuffers(1,&vboBG);
    glBindVertexArray(vaoBG);
    glBindBuffer(GL_ARRAY_BUFFER,vboBG);
    glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,camW,camH,0,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
    GLint uTex = glGetUniformLocation(progBG,"uTex");

    GLuint progGeom = makeProgram(VS_GEOM, FS_GEOM, "aPos");
    GLint uP  = glGetUniformLocation(progGeom,"uP");
    GLint uMV = glGetUniformLocation(progGeom,"uMV");
    GLint uC  = glGetUniformLocation(progGeom,"uColor");

    GLuint vaoBox=0, vboBox=0, eboBox=0;
    {
        const float verts[] = {
            0.f,-0.5f,-0.5f,  1.f,-0.5f,-0.5f,
            1.f, 0.5f,-0.5f,  0.f, 0.5f,-0.5f,
            0.f,-0.5f, 0.5f,  1.f,-0.5f, 0.5f,
            1.f, 0.5f, 0.5f,  0.f, 0.5f, 0.5f
        };
        const unsigned int idx[] = {
            4,5,6, 4,6,7,
            0,1,2, 0,2,3,
            0,4,7, 0,7,3,
            1,5,6, 1,6,2,
            0,1,5, 0,5,4,
            3,2,6, 3,6,7
        };
        glGenVertexArrays(1,&vaoBox);
        glGenBuffers(1,&vboBox);
        glGenBuffers(1,&eboBox);
        glBindVertexArray(vaoBox);
        glBindBuffer(GL_ARRAY_BUFFER,vboBox);
        glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,eboBox);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idx),idx,GL_STATIC_DRAW);
        glBindVertexArray(0);
    }

    cv::Matx44f P = projFromK(K, camW, camH, 0.01f, 100.0f);
    float Pgl[16]; toGLColumnMajor(P, Pgl);

    ArucoTracker tracker(MARKER_LEN, K, dist);
    std::cout << "tracker initialized\n";

    glClearColor(0,0,0,1);

    {
        auto model = OBJLoader::load("../models/tower.obj"); 
        if (model.vertices.empty() || model.indices.empty()) {
            std::cerr << "Failed to load OBJ Model\n";
        } else {
            float xmin= std::numeric_limits<float>::max(), ymin= xmin, zmin= xmin;
            float xmax=-xmin, ymax=-xmin, zmax=-xmin;
            for (const auto& v : model.vertices) {
                xmin = std::min(xmin, v.x); xmax = std::max(xmax, v.x);
                ymin = std::min(ymin, v.y); ymax = std::max(ymax, v.y);
                zmin = std::min(zmin, v.z); zmax = std::max(zmax, v.z);
            }
            const float dx = xmax - xmin, dy = ymax - ymin;
            const float cx = 0.5f * (xmin + xmax);
            const float cy = 0.5f * (ymin + ymax);
            const float footprint = 0.8f * MARKER_LEN;
            const float s = footprint / std::max(dx, dy);

            std::vector<float> packed; packed.reserve(model.vertices.size() * 3);
            for (const auto& v : model.vertices) {
                packed.push_back((v.x - cx) * s);
                packed.push_back((v.y - cy) * s);
                packed.push_back((v.z - zmin) * s); 
            }

            glGenVertexArrays(1, &vaoModel);
            glGenBuffers(1, &vboModel);
            glGenBuffers(1, &eboModel);

            glBindVertexArray(vaoModel);
            glBindBuffer(GL_ARRAY_BUFFER, vboModel);
            glBufferData(GL_ARRAY_BUFFER, packed.size() * sizeof(float), packed.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboModel);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, model.indices.size() * sizeof(uint32_t),
                         model.indices.data(), GL_STATIC_DRAW);
            glBindVertexArray(0);

            idxCount   = static_cast<GLsizei>(model.indices.size());
            modelReady = true;
            std::cout << "Model ready: " << idxCount << " indices\n";
        }
    }

    while(!glfwWindowShouldClose(win)) {
        cv::Mat bgr; cap >> bgr; if(bgr.empty()) break;
        cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        if(rgb.cols!=camW || rgb.rows!=camH){
            camW=rgb.cols; camH=rgb.rows;
            glBindTexture(GL_TEXTURE_2D,tex);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,camW,camH,0,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
            P = projFromK(K, camW, camH, 0.01f, 100.0f);
            toGLColumnMajor(P, Pgl);
        }

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,camW,camH,GL_RGB,GL_UNSIGNED_BYTE,rgb.data);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glUseProgram(progBG);
        glUniform1i(uTex,0);
        glBindVertexArray(vaoBG);
        glDrawArrays(GL_TRIANGLE_STRIP,0,4);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        cv::Vec3d rvec, tvec;
        bool x = tracker.detect(bgr, rvec, tvec);
        if (x) {
            cv::Mat Rcv; cv::Rodrigues(rvec, Rcv);
            cv::Matx44f M = cv::Matx44f::eye();
            for(int i=0;i<3;i++) for(int j=0;j<3;j++) M(i,j) = float(Rcv.at<double>(i,j));
            M(0,3) = float(tvec[0]); M(1,3) = float(tvec[1]); M(2,3) = float(tvec[2]);
            cv::Matx44f CvToGl = cv::Matx44f::eye(); CvToGl(1,1)=-1.f; CvToGl(2,2)=-1.f;
            cv::Matx44f MV = mul44(CvToGl, M);

            if (showModel && modelReady) {
                const float eps = 0.002f * MARKER_LEN; // lift a hair to avoid z-fighting
                cv::Matx44f MV_final = mul44(MV, makeTranslate(0.f, 0.f, eps));
                float MVgl[16]; toGLColumnMajor(MV_final, MVgl);

                glUseProgram(progGeom);
                glUniformMatrix4fv(uP,  1, GL_FALSE, Pgl);
                glUniformMatrix4fv(uMV, 1, GL_FALSE, MVgl);
                glUniform3f(uC, 1.f, 1.f, 1.f); // solid white

                glBindVertexArray(vaoModel);
                glDrawElements(GL_TRIANGLES, idxCount, GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            } else {
                glUseProgram(progGeom);
                glUniformMatrix4fv(uP,  1, GL_FALSE, Pgl);

                const float L  = MARKER_LEN;
                const float th = L * 0.06f;

                {
                    cv::Matx44f MVx = mul44(MV, mul44(makeTranslate((L - th)*0.5f, 0.f, 0.f), makeScale(L + th, th, th)));
                    float MVxgl[16]; toGLColumnMajor(MVx, MVxgl);
                    glUniformMatrix4fv(uMV, 1, GL_FALSE, MVxgl);
                    glUniform3f(uC, 1,0,0);
                    glBindVertexArray(vaoBox);
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
                }
                {
                    cv::Matx44f MVy = mul44(MV, mul44(makeTranslate(0.f, L*0.5f, 0.f), makeScale(th, L, th)));
                    float MVygl[16]; toGLColumnMajor(MVy, MVygl);
                    glUniformMatrix4fv(uMV, 1, GL_FALSE, MVygl);
                    glUniform3f(uC, 0,1,0);
                    glBindVertexArray(vaoBox);
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
                }
                {
                    cv::Matx44f MVz = mul44(MV, mul44(makeTranslate(0.f, 0.f, L*0.5f), makeScale(th, th, L)));
                    float MVzgl[16]; toGLColumnMajor(MVz, MVzgl);
                    glUniformMatrix4fv(uMV, 1, GL_FALSE, MVzgl);
                    glUniform3f(uC, 0,0,1);
                    glBindVertexArray(vaoBox);
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
                }
                glBindVertexArray(0);
            }
        }

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
