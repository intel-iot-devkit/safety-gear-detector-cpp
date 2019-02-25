/*
 * Copyright (c) 2018 Intel Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <opencv2/imgproc.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/pvl.hpp>

typedef struct
{
    std::string name;
    cv::Rect rect;
    bool jacket;
    bool hardHat;
} workerAttr;

class Video {
public:
    std::string name;

    cv::VideoCapture video;
    int width;
    int height;
    bool isCam = false;

    std::map<int, workerAttr> detectedWorkers;
    int violations = 0;
    int totalPeople = 0;

    int frames = 0;
    int totalFrames = 0;
    
    double FPS;
    std::chrono::high_resolution_clock::time_point loop_start_time;
    std::chrono::high_resolution_clock::time_point loop_end_time;

    Video(){};

    Video(const int idx,
            const std::string path)
            {
                if (std::all_of(path.begin(), path.end(), ::isdigit))
                {
                    video = cv::VideoCapture(std::stoi(path));
                    isCam = true;
                    name = std::string("Cam " + std::to_string(idx));
                }
                else
                {
                    std::ifstream f(path);
                    if (f.good())
                    {
                        f.close();
                        video = cv::VideoCapture(path);
                        name = std::string("Video " + std::to_string(idx));
                        totalFrames = video.get(cv::CAP_PROP_FRAME_COUNT);
                    }
                    else
                    {
                        std::cout << "File " << path << " doesn't exist.\n";
                        exit(21);
                    }
                }
                if (!video.isOpened())
                {
                    std::cout << "Couldn't open video " << path << std::endl;
                    exit(20);
                }
                height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
                width = video.get(cv::CAP_PROP_FRAME_WIDTH);

                loop_start_time = std::chrono::high_resolution_clock::now();
            }
};
