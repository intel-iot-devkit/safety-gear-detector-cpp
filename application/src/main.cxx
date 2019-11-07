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
#include <chrono>
#include "inference.hpp"
#include "global_vars.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <nlohmann/json.hpp>

using namespace InferenceEngine::details;
using namespace InferenceEngine;
using namespace std;
using json = nlohmann::json;
json jsonobj;

bool SAFETY_MODEL = false;

// Parse the configuration file config.json to get the videos
std::vector<Video> getVideos()
{
    std::vector<Video> videos;
    std::string vid, conf_file = "../resources/config.json";
    std::ifstream confFile(conf_file);
    if (!confFile.is_open())
    {
        std::cout << "Could not open config file" << std::endl;
        exit(0);
    }
    confFile>>jsonobj;
    int cnt = 0;
    auto obj = jsonobj["inputs"];
    for(int i=0;i<obj.size();i++)
    {
        auto vid = obj[i]["video"];
        videos.emplace_back(i, vid);
    }
    return videos;
}

/*void trainFaceRec(cv::Ptr<cv::pvl::FaceDetector> fd, cv::Ptr<cv::pvl::FaceRecognizer> fr)
{
    int personID = -1;
    std::vector<cv::pvl::Face> detectedFaces;

    std::ifstream db(conf_workersFile);
    std::string label;
    std::string img;
    std::string lbl = "";
    int faces = 0;

    while (std::getline(db, label, ';'))
    {
        std::getline(db, img);

        detectedFaces.clear();

        cv::Mat imgIn = cv::imread(img);
        cv::Mat imgGray;
        cvtColor(imgIn, imgGray, cv::COLOR_BGR2GRAY);

        fd->detectFaceRect(imgGray, detectedFaces);

        if (detectedFaces.size() == 0)
        {
            std::cout << "No faces were detected in image " << img << std::endl;
            continue;
        }
        else if (detectedFaces.size() > 1)
        {
            std::cout << "More than one face detected in image " << img << ". Only images with one face can be used for training. Ignored!\n";

            for (size_t i = 0; i < detectedFaces.size(); i++)
            {
                cv::rectangle(imgIn, detectedFaces[i].get<cv::Rect>(cv::pvl::Face::FACE_RECT), cv::Scalar(0, 255, 0));
            }
            continue;
        }

        if (label == lbl)
        {
            if (faces >= 9)
            {
                continue;
            }
        }
        else
        {
            lbl = label;
            personID = fr->createNewPersonID();
            workerNames.emplace(personID, label);
            faces = 0;

            // DEBUG
            std::cout << "Added " << label << "\n";
        }

        fr->registerFace(imgGray, detectedFaces[0], personID);
        faces++;
    }

    //NOTE!: save is fubar
    // std::cout << "Saving DB to " << conf_dbPath << std::endl;
    // fr->save(conf_dbPath);

    return;
}
*/


/*
// Detect the Hat
bool detectHardHat(cv::Mat &img)
{
    cv::Range rng(0, img.rows / 10);
    cv::Mat inImg(img, rng);
    cv::Mat hsv;

    int lowH = 22;
    int lowS = 65;
    int lowV = 150;

    int highH = 28;
    int highS = 255;
    int highV = 255;

    int crop = 0;
    int height = 15;
    int perc = 8;

    cv::cvtColor(inImg, hsv, cv::COLOR_BGR2HSV);

    cv::Mat thrImg;
    cv::inRange(hsv, cv::Scalar(lowH, lowS, lowV), cv::Scalar(highH, highS, highV), thrImg);

    cv::Rect roi;
    roi.x = 0;
    roi.y = thrImg.rows * crop / 100;
    roi.width = thrImg.cols;
    roi.height = thrImg.rows * height / 100;
    cv::Mat imgCropped(thrImg, roi);

    if (cv::countNonZero(thrImg) < imgCropped.total() * perc / 100)
    {
        return false;
    }

    return true;
}

// Detect the safety jacket
bool detectSafetyJacket(cv::Mat &img)
{

    int chunk = img.rows / 10;
    cv::Range rng(chunk * 2, chunk * 7);
    cv::Mat inImg(img, rng);
    cv::Mat hsv;

    int lowH = 4;
    int lowS = 150;
    int lowV = 60;

    int highH = 8;
    int highS = 255;
    int highV = 255;

    int crop = 15;
    int height = 40;
    int perc = 25;

    cv::cvtColor(inImg, hsv, cv::COLOR_BGR2HSV);

    cv::Mat thrImg;
    cv::inRange(hsv, cv::Scalar(lowH, lowS, lowV), cv::Scalar(highH, highS, highV), thrImg);

    cv::Rect roi;
    roi.x = 0;
    roi.y = thrImg.rows * crop / 100;
    roi.width = thrImg.cols;
    roi.height = thrImg.rows * height / 100;
    cv::Mat imgCropped(thrImg, roi);

    if (cv::countNonZero(thrImg) < imgCropped.total() * perc / 100)
    {
        return false;
    }

    return true;
}


void recognizeWorker_model(cv::Ptr<cv::pvl::FaceDetector> fd, cv::Ptr<cv::pvl::FaceRecognizer> fr, cv::Mat &img, std::vector<workerAttr> &inFrameWorkers, Video *vid)
{
    fd->setTrackingModeEnabled(true);
    cv::Mat grayImg;

    for (int i = 0; i < inFrameWorkers.size(); i++)
    {
        std::vector<cv::pvl::Face> detectedFaces;
        std::vector<int> personIDs;
        std::vector<int> confidence;

        cv::Mat crop;
        try
        {
            crop = cv::Mat(img, inFrameWorkers[i].rect);
        }
        catch (cv::Exception e)
        {
            std::cout << "Caught exception: " << e.what();
            std::cout << img.cols << ' ' << inFrameWorkers[i].rect.width << ' ' << img.rows << ' ' << inFrameWorkers[i].rect.height << '\n';
        }
        cv::cvtColor(crop, grayImg, cv::COLOR_BGR2GRAY);

        try
        {
            fd->detectFaceRect(grayImg, detectedFaces);
        }
        catch (cv::Exception e)
        {
            std::cout << "Caught exception: " << e.what();
        }

        if (detectedFaces.size() == 1)
        {
            
            fr->recognize(grayImg, detectedFaces, personIDs, confidence);
            auto workerName = workerNames.find(personIDs[0]);
            if (workerName != workerNames.end())
            {
                auto worker = (*vid).detectedWorkers.find(personIDs[0]);
                if (confidence[0] >= conf_faceRecogConfidenceThreshold)
                {
                    char tmstmp[10];
                    inFrameWorkers[i].name = workerName->second;
                    if (worker == (*vid).detectedWorkers.end())
                    {
                        (*vid).detectedWorkers.insert(std::pair<int, workerAttr>(personIDs[0], inFrameWorkers[i]));

                    }              
                    (*vid).detectedWorkers[personIDs[0]].count++;

                    if (inFrameWorkers[i].jacket)
                    {
                        (*vid).detectedWorkers[personIDs[0]].jacketCount++;
                    }

                    if (inFrameWorkers[i].hardHat)
                    {
                        (*vid).detectedWorkers[personIDs[0]].hatCount++;
                    }

                    time_t t = time(nullptr);
                    tm *currTime = localtime(&t);
                    sprintf(tmstmp, "%02d:%02d:%02d", currTime->tm_hour, currTime->tm_min, currTime->tm_sec);
 
                    if((*vid).detectedWorkers[personIDs[0]].count == 5)
                    {
                        if ((*vid).detectedWorkers[personIDs[0]].jacketCount > 3 && (*vid).detectedWorkers[personIDs[0]].hatCount > 3)
                        {
                            std::cout << tmstmp << " - " << (*vid).name << ": " << inFrameWorkers[i].name << " is fully equipped\n";
                        }                   
                        else
                        {
                            std::string hardHat;
                            std::string jacket;
                            if ((*vid).detectedWorkers[personIDs[0]].jacketCount > 3)
                            {
                                jacket = "";
                            }
                            else
                            {
                                jacket = "not ";
                            }

                            if ((*vid).detectedWorkers[personIDs[0]].hatCount > 3)
                            {
                                hardHat = "";
                            }
                            else
                            {
                                hardHat = "not ";
                            }
                            std::cout << tmstmp << " - " << (*vid).name << ": " << inFrameWorkers[i].name << " is " << hardHat << "wearing hard hat and "
                                        << jacket << "wearing safety jacket\n";

                            ++(*vid).violations;
                        }
                        ++(*vid).totalPeople;
                    }


                }
            }
        }
    }
}


void recognizeWorker(cv::Ptr<cv::pvl::FaceDetector> fd, cv::Ptr<cv::pvl::FaceRecognizer> fr, cv::Mat &img, std::vector<workerAttr> &inFrameWorkers, Video *vid)
{
    fd->setTrackingModeEnabled(true);

    cv::Mat grayImg;

    for (int i = 0; i < inFrameWorkers.size(); i++)
    {
        std::vector<cv::pvl::Face> detectedFaces;
        std::vector<int> personIDs;
        std::vector<int> confidence;

        cv::Mat crop;
        try
        {
            crop = cv::Mat(img, inFrameWorkers[i].rect);
        }
        catch (cv::Exception e)
        {
            std::cout << "Caught exception: " << e.what();
            std::cout << img.cols << ' ' << inFrameWorkers[i].rect.width << ' ' << img.rows << ' ' << inFrameWorkers[i].rect.height << '\n';
        }
        cv::cvtColor(crop, grayImg, cv::COLOR_BGR2GRAY);

        try
        {
            fd->detectFaceRect(grayImg, detectedFaces);
        }
        catch (cv::Exception e)
        {
            std::cout << "Caught exception: " << e.what();
        }

        if (detectedFaces.size() == 1)
        {
            fr->recognize(grayImg, detectedFaces, personIDs, confidence);
            auto workerName = workerNames.find(personIDs[0]);
            if (workerName != workerNames.end())
            {
                auto worker = (*vid).detectedWorkers.find(personIDs[0]);
                if (worker == (*vid).detectedWorkers.end() && confidence[0] >= conf_faceRecogConfidenceThreshold)
                {
                    inFrameWorkers[i].name = workerName->second;
                    inFrameWorkers[i].jacket = detectSafetyJacket(crop);
                    std::string jacket;
                    if (inFrameWorkers[i].jacket)
                    {
                        jacket = "";
                    }
                    else
                    {
                        jacket = "not ";
                    }
                    inFrameWorkers[i].hardHat = detectHardHat(crop);
                    std::string hardHat;
                    if (inFrameWorkers[i].hardHat)
                    {
                        hardHat = "";
                    }
                    else
                    {
                        hardHat = "not ";
                    }

                    char tmstmp[10];
                    time_t t = time(nullptr);
                    tm *currTime = localtime(&t);
                    sprintf(tmstmp, "%02d:%02d:%02d", currTime->tm_hour, currTime->tm_min, currTime->tm_sec);

                    if (inFrameWorkers[i].hardHat && inFrameWorkers[i].jacket)
                    {
                        std::cout << tmstmp << " - " << (*vid).name << ": " << inFrameWorkers[i].name << " is fully equipped\n";
                    }
                    else
                    {
                        std::cout << tmstmp << " - " << (*vid).name << ": " << inFrameWorkers[i].name << " is " << hardHat << "wearing hard hat and "
                                  << jacket << "wearing safety jacket\n";

                        ++(*vid).violations;
                    }

                    (*vid).detectedWorkers.insert(std::pair<int, workerAttr>(personIDs[0], inFrameWorkers[i]));
                    ++(*vid).totalPeople;
                }
            }
        }
    }
}
*/


int main(int argc, char const *argv[])
{
    std::string conf_modelLayers;
    std::string conf_modelWeights;
    std::string conf_modelLayers_hat;
    std::string conf_modelWeights_hat;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    Network net;

    Network net_hat;
    bool isAsyncmode = true;
    std::vector<bool> noMoreData;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        exit(0);
    }
    if (parser.has("device"))
    {
        myTargetDevice = parser.get<cv::String>("device");
        std::string dummy = myTargetDevice;
        if(!(myTargetDevice.find("MULTI:") != std::string::npos))
        {
            if (!(std::find(acceptedDevices.begin(), acceptedDevices.end(), myTargetDevice) != acceptedDevices.end()))
            {
                std::cout << "Unsupported device " << myTargetDevice << std::endl;
                exit(10);
            }
        }
    }
    else
    {
        myTargetDevice = "CPU";
    }
    /*
    if (parser.has("database"))
    {
        conf_workersFile = parser.get<cv::String>("database");
    }
    else
    {
        std::cout << "Please specify path to worker database\n";
    }
    */

    std::vector<Video> videos;
    videos = getVideos();

    if (parser.has("loop"))
    {
        loop = parser.get<bool>("loop");
    }

    if (parser.has("model"))
    {
        conf_modelLayers = parser.get<cv::String>("model");
        int pos = conf_modelLayers.rfind(".");
        conf_modelWeights = conf_modelLayers.substr(0, pos) + ".bin";
        if (myTargetDevice.find("CPU") != std::string::npos)
    	{
            net.plugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), "CPU");
    	}
        if (net.loadNetwork(conf_modelLayers, conf_modelWeights, net.plugin, myTargetDevice) != 0)
            return EXIT_FAILURE;

    }
    else
    {
        std::cout << "Please specify xml model path for person detection.\n";
        exit(1);
    }

    if (parser.has("model_hat"))
    {
        conf_modelLayers_hat = parser.get<cv::String>("model_hat");
        int pos = conf_modelLayers_hat.rfind(".");
        conf_modelWeights_hat = conf_modelLayers_hat.substr(0, pos) + ".bin";
        net_hat.loadNetwork(conf_modelLayers_hat, conf_modelWeights_hat, net.plugin, myTargetDevice);
        SAFETY_MODEL = true;
    }
    else
    {
        SAFETY_MODEL = false;
    }
    if (parser.has("flag"))
    {
        if(parser.get<cv::String>("model_hat") == "async")
        {
            net.isAsync = 1;
            net_hat.isAsync = 1;
            isAsyncmode = true;
            std::cout<<"Application running in async mode"<<std::endl;
        }
        else
        {
            net.isAsync = 0;
            net_hat.isAsync = 0;
            isAsyncmode = false;
            std::cout<<"Application running in sync mode"<<std::endl;
        }
    }
    else
    {
        net.isAsync = 1;
        net_hat.isAsync = 1;
        isAsyncmode = true;
        std::cout<<"Application running in async mode"<<std::endl;
    }
    const size_t modelWidth = net.getModelWidth();
    const size_t modelHeight = net.getModelHeight();
    const size_t modelWidth_hat = net_hat.getModelWidth();
    const size_t modelHeight_hat = net_hat.getModelHeight();

    // Init PVL
    /* cv::Ptr<cv::pvl::FaceDetector> faceDet = cv::pvl::FaceDetector::create();
    cv::Ptr<cv::pvl::FaceRecognizer> faceRec = cv::pvl::FaceRecognizer::create();

    trainFaceRec(faceDet, faceRec);

    std::cout << "Worker Names: ";
    for (auto &&i : workerNames)
    {
        std::cout << i.second << ' ';
    }
    std::cout << "\n";
    */
    // Main loop
    int minFPS = 30;
    int index = 0;

    for (auto &&i : videos)
    {
        minFPS = std::min(minFPS, (int)round(i.video.get(cv::CAP_PROP_FPS)));
        noMoreData.push_back(false);
        cv::namedWindow(i.name, cv::WINDOW_NORMAL);
    }
    std::chrono::high_resolution_clock::time_point loop_start_time = std::chrono::high_resolution_clock::now();
    Video *prevVideo;
    int violations = 0, conf_inFrameViolationsThreshold = 19, people = 0;
    cv::Mat prev_frame = cv::Mat::zeros(100,100,CV_8UC1);
    while (true)
    {
        index = 0;
        for (auto &&currVideo : videos)
        {
            cv::Mat currImg;
            std::vector<cv::Rect> detections;
            std::vector<workerAttr> inFrameWorkers;
            currVideo.video >> currImg;
            currVideo.frame = currImg.clone();
            currVideo.frames++;
            if (currImg.empty())
            {
                noMoreData[index] = true;
                ++index;
                cv::Mat messageWindow = cv::Mat(currVideo.height, currVideo.width, CV_8UC1, cv::Scalar(0));
                std::string message = "Video stream from " + currVideo.name + " has ended!";
                cv::putText(messageWindow, message, cv::Point(160, currVideo.height / 2),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1);
                cv::imshow(currVideo.name, messageWindow);
                continue;
            }

            if(!isAsyncmode)
            {
                prevVideo = &currVideo;
                prev_frame = currVideo.frame;
            }

            // Resize image
            cv::Mat rsImg, rsImg_hat, or_hat;
            cv::resize(currImg, rsImg, cv::Size(modelWidth, modelHeight));
            net.fillInputBlob(rsImg);
            std::chrono::high_resolution_clock::time_point infer_start_time = std::chrono::high_resolution_clock::now();
            net.inferenceRequest();
            std::chrono::high_resolution_clock::time_point infer_end_time = std::chrono::high_resolution_clock::now();
            ms infer_time = std::chrono::duration_cast<ms>(infer_end_time - infer_start_time);
            people = 0;
            if(net.wait() == 0)
            {
		    // Get inference results
		    float *results = net.inference();

		    // Detected objects in the current frame
		    for (int i = 0; i < net.maxProposalCount; i++)
		    {
		        workerAttr wk;
		        wk.hardHat = false;
		        wk.jacket = false;

		        float *result = results + i * net.objectSize;
		        float imageID = result[0];
		        int label = static_cast<int>(result[1]);
		        float confidence = result[2];

		        if (label == conf_modelPersonLabel && confidence >= conf_inferConfidenceThreshold)
		        {
		            // Replace modelWidth/modelHeight with original image values if you show the original image
		            float xmin = result[3] * prevVideo->width;
		            float ymin = result[4] * prevVideo->height;
		            float xmax = result[5] * prevVideo->width;
		            float ymax = result[6] * prevVideo->height;
		            int padding = 30;
		            ymin = (ymin - padding) > 0 ? (ymin - padding) : 0;
		            xmin = (xmin - padding) > 0 ? (xmin - padding) : 0;
		            ymax = (ymax + padding) < prevVideo->height ? (ymax + padding) : prevVideo->height;
		            xmax = (xmax + padding) < prevVideo->width ? (xmax + padding) : prevVideo->width;
		            people++;
		            wk.rect = cv::Rect(cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax));
		            or_hat = prev_frame(wk.rect);
                            inFrameWorkers.push_back(wk);
		            if (SAFETY_MODEL)
		            {
		                cv::resize(or_hat, rsImg_hat, cv::Size(modelWidth_hat, modelHeight_hat));
		                net_hat.fillInputBlob(rsImg_hat);
		                std::chrono::high_resolution_clock::time_point infer_start_time = std::chrono::high_resolution_clock::now();

		                net_hat.inferenceRequest();
		                net_hat.wait();
		                // Get inference results
		                float *results1 = net_hat.inference();
		                violations =0;
		                for (int i_hat = 0; i_hat < net_hat.maxProposalCount; i_hat++)
		                {
		                    float *result1 = results1 + i_hat * net_hat.objectSize;
		                    float imageID1 = result1[0];
		                    int label1 = static_cast<int>(result1[1]);
		                    float confidence1 = result1[2];
		                    if (label1 == 2 && confidence1 >= 0.4)
		                    {
		                        float xmin1 = result1[3] * or_hat.cols;
		                        float ymin1 = result1[4] * or_hat.rows;
		                        float xmax1 = result1[5] * or_hat.cols;
		                        float ymax1 = result1[6] * or_hat.rows;
		                        if(wk.jacket == false)
		                        {
		                            detections.push_back(cv::Rect(cv::Point(xmin1 + xmin, ymin1 + ymin), cv::Point(xmax1 + xmin, ymax1 + ymin)));
		                            wk.jacket = true;
		                        }
		                    }
		                    if (label1 == 4 && confidence1 >= 0.4)
		                    {
		                        float xmin1 = result1[3] * or_hat.cols;
		                        float ymin1 = result1[4] * or_hat.rows;
		                        float xmax1 = result1[5] * or_hat.cols;
		                        float ymax1 = result1[6] * or_hat.rows;
		                        if(wk.hardHat == false)
		                        {
		                            detections.push_back(cv::Rect(cv::Point(xmin1 + xmin, ymin1 + ymin), cv::Point(xmax1 + xmin, ymax1 + ymin)));
		                            wk.hardHat = true;
		                        }
		                    }
		                }
		                if(wk.hardHat == false || wk.jacket == false)
		                {
		                    violations++;
		                }
		                if(isAsyncmode)
		                    net_hat.swapInferenceRequest();
		            }
                            inFrameWorkers.push_back(wk);
		        }
		    }
		    if(violations == prevVideo->currentViolationCount)
		    {
		        prevVideo->currentViolationCountConfidence += 1;
		        if( prevVideo->currentViolationCountConfidence == conf_inFrameViolationsThreshold)
		        {
		            if(prevVideo->currentViolationCount > prevVideo->prevViolationCount)
		            {
		                prevVideo->totalViolations += prevVideo->currentViolationCount - prevVideo->prevViolationCount;
		            }
		            prevVideo->prevViolationCount = prevVideo->currentViolationCount;
		        }
		    }
		    else
		    {
		        prevVideo->currentViolationCountConfidence = 0;
		        prevVideo->currentViolationCount = violations;
		    }
		    if(people == prevVideo->currentPeopleCount)
		    {
		        prevVideo->currentPeopleCountConfidence += 1;
		        if(prevVideo->currentPeopleCountConfidence == conf_inFrameViolationsThreshold)
		        {
		            prevVideo->currentTotalPeopleCount += prevVideo->currentPeopleCount - prevVideo->prevPeopleCount;
		            if(prevVideo->currentTotalPeopleCount > prevVideo->prevPeopleCount)
		            {
		                prevVideo->totalPeopleCount += prevVideo->currentTotalPeopleCount - prevVideo->prevPeopleCount;
		            }
		            prevVideo->prevPeopleCount = prevVideo->currentPeopleCount;
		        }
		    }
		    else
		    {
		            prevVideo->currentPeopleCountConfidence = 0;
		            prevVideo->currentPeopleCount = people;
		    }

		    /*if (SAFETY_MODEL)
		    {

		        recognizeWorker_model(faceDet, faceRec, currImg, inFrameWorkers, &currVideo);
		    }
		    else
		    {
		        recognizeWorker(faceDet, faceRec, currImg, inFrameWorkers, &currVideo);
		    }
		    */
		    for (auto &&wk : inFrameWorkers)
		    {
		        cv::rectangle(prev_frame, wk.rect, cv::Scalar(0, 255, 0), 2);
		    }
		    if (SAFETY_MODEL)
		    for (auto _rect : detections)
		    {
		        cv::rectangle(prev_frame, _rect, cv::Scalar(0, 255, 0), 2);
		    }
		    std::chrono::high_resolution_clock::time_point loop_end_time = std::chrono::high_resolution_clock::now();
		    prevVideo->loop_end_time = std::chrono::high_resolution_clock::now();
		    std::chrono::duration<float> frame_time = std::chrono::duration_cast<std::chrono::duration<float>>(prevVideo->loop_end_time - prevVideo->loop_start_time);
		    char txt[50];
		    sprintf(txt, "Total worker count: %d", prevVideo->totalPeopleCount);
		    cv::putText(prev_frame, txt, cv::Point(10, prevVideo->height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		    sprintf(txt, "Total violations count: %d", prevVideo->totalViolations);
		    cv::putText(prev_frame, txt, cv::Point(10, prevVideo->height - 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		    sprintf(txt, "FPS: %.2f", 1 / frame_time.count());
		    cv::putText(prev_frame, txt, cv::Point(10, prevVideo->height - 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

                    if (!isAsyncmode)
                    {
			sprintf(txt, "Infer time: %.3f ms", 1000*infer_time.count());
                    }
                    else
			sprintf(txt, "Infer time: N/A for Async mode");
		    cv::putText(prev_frame, txt, cv::Point(10, prevVideo->height - 70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		    cv::imshow(prevVideo->name, prev_frame);
		    loop_start_time = std::chrono::high_resolution_clock::now();
		    prevVideo->loop_start_time = std::chrono::high_resolution_clock::now();

		    // Press Esc to exit the application
		    if (cv::waitKey(1) == 27)
		    {
		        net.currInfReq = NULL;
		        return 0;
		    }

		    if (loop && !currVideo.isCam && currVideo.frames == currVideo.totalFrames)
		    {
		        currVideo.frames = 0;
		        currVideo.video.set(cv::CAP_PROP_POS_FRAMES, 0);
		        currVideo.detectedWorkers.clear();
		    }
		    inFrameWorkers.clear();
		    ++frames;
		    ++index;
		    detections.clear();
            }
            if(isAsyncmode)
            {
                prevVideo = &currVideo;
                net.swapInferenceRequest();
                prev_frame = currVideo.frame.clone();
            }
        }

        // Check if all the videos have ended
        if (find(noMoreData.begin(), noMoreData.end(), false) == noMoreData.end())
            break;
    }
    net.currInfReq = NULL;
    cv::destroyAllWindows();
    return 0;
}
