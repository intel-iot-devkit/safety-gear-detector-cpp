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

#pragma once

#include <vector>
#include <utility>
#include <string>
#include <map>

#include "structs.hpp"


static std::vector<std::string> acceptedDevices{"CPU", "GPU", "MYRIAD", "HETERO:FPGA,CPU", "HDDL"};

std::map<int, std::string> workerNames;
std::string conf_workersFile;
std::string conf_dbPath;
float conf_inferConfidenceThreshold = 0.7;
int conf_faceRecogConfidenceThreshold = -65536;
int conf_modelPersonLabel = 1;

bool loop = false;

const cv::String keys = 
    "{help h        |   | prints this message}"
    "{m model       |   |  Path to an .xml file with a trained model}"
    "{mh model_hat|   |  Path to an .xml file with a trained HatandVest model}"
    "{d device      |   | Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or FPGA is acceptable. Application "
                                "will look for a suitable plugin for device specified (CPU by default)}"
    "{lp loop       |   | Loops video input}"
    "{f flag        |   | Execution on SYNC or ASYNC mode. Default option is ASYNC mode}"
    ;

//DEBUG
int frames = -1;
