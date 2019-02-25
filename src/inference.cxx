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

#include <string>
#include "inference.hpp"

std::string myTargetDevice;
std::string conf_modelLayers;
std::string conf_modelWeights;


// Default Constructor
Network::Network()
{    
    modelHeight = 0;
    modelWidth = 0;
    maxProposalCount = -1;
    conf_batchSize = 1;
    statusCodeOK = InferenceEngine::StatusCode::OK;
}

// Load the plugin and configure the network 
int Network::loadNetwork()
{
    // Init plugin
    InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher({""}).getPluginByDevice(myTargetDevice);

    // Add CPU extensions if necessary
    if (myTargetDevice == "CPU")
    {
        plugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>());
    }

    // Configure network
    InferenceEngine::CNNNetReader networkReader;
    networkReader.ReadNetwork(conf_modelLayers);
    networkReader.ReadWeights(conf_modelWeights);
    networkReader.getNetwork().setBatchSize(conf_batchSize);

    // Get input info
    inputInfo = new InferenceEngine::InputsDataMap(networkReader.getNetwork().getInputsInfo());
    if (inputInfo->size() != 1)
    {
        std::cout << "This application only supports networks with one input\n";
        return -1;
    }
    auto inputName = inputInfo->begin()->first;

    InferenceEngine::SizeVector inputDims = inputInfo->begin()->second->getDims();
    if (inputDims.size() != 4)
    {
        std::cout << "Not supported input dimensions size, expected 4, got "
                  << inputDims.size() << std::endl;
    }
    modelWidth = inputDims[0];
    modelHeight = inputDims[1];
    const size_t modelChannels = inputDims[2];

    auto channelSize = modelHeight * modelWidth;
    auto inputSize = channelSize * modelChannels;

    // Set input info
    (*inputInfo)[inputName]->setInputPrecision(InferenceEngine::Precision::U8);
    (*inputInfo)[inputName]->setLayout(InferenceEngine::Layout::NCHW);
    // !NOTE! Alternative
    // InferenceEngine::InputInfo::Ptr &input = inputInfo.begin()->second;
    // input->setPrecision(InferenceEngine::Precision::U8);
    // input->getInputData()->setLayout(Layout::NCHW);

    // Get output info
    InferenceEngine::OutputsDataMap outputInfo(networkReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 1)
    {
        std::cout << "This application only supports networks with one output\n";
        return -1;
    }

    outputName = outputInfo.begin()->first;
    InferenceEngine::DataPtr &output = outputInfo.begin()->second;
    //const int num_classes = networkReader.getNetwork().getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
    InferenceEngine::SizeVector outputDims = output->getDims();
    maxProposalCount = outputDims[2]; // SSD detected objects
    objectSize = outputDims[3];       // SSD output per object

    // Set output info
    output->setPrecision(InferenceEngine::Precision::FP32);
    output->setLayout(InferenceEngine::Layout::NCHW);

    // Load model into plugin
    InferenceEngine::ExecutableNetwork network = plugin.LoadNetwork(networkReader.getNetwork(), {});

    // Create inference requests
    currInfReq = network.CreateInferRequestPtr();
    nextInfReq = network.CreateInferRequestPtr();
    return 0;
}

// Transfer data from OpenCV Mat to InferRequest Blob
template <typename T>
void Network::cvMatToBlob(const cv::Mat &img, InferenceEngine::Blob::Ptr &blob)
{
    // Get Blob info
    InferenceEngine::SizeVector blobSize = blob.get()->getTensorDesc().getDims();
    size_t channels = blobSize[1];
    size_t height = blobSize[2];
    size_t width = blobSize[3];
    auto resolution = height * width;

    // Get pointer to blob data as type T
    T *blobData = blob->buffer().as<T *>();

    // Fill blob
    for (size_t c = 0; c < channels; c++)
    {
        auto aux = c * resolution;
        for (size_t h = 0; h < height; h++)
        {
            auto aux2 = h * width;
            for (size_t w = 0; w < width; w++)
            {
                blobData[aux + aux2 + w] = img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }

    return;
}

size_t Network::getModelHeight()
{
    return modelHeight;
}

size_t Network::getModelWidth()
{
    return modelWidth;
}

// Fill Input Blob
void Network::fillInputBlob(cv::Mat img)
{
    auto inputBlob = nextInfReq->GetBlob(inputInfo->begin()->first);
    cvMatToBlob<uchar>(img, inputBlob);
}

// Create inference request
void Network::inferenceRequest()
{
    // Request async infer
    nextInfReq->StartAsync();

    // Check if previous infer request is ready
    stsCd = currInfReq->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

// Get the inference output
float *Network::inference()
{
    return currInfReq->GetBlob(outputName)->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
}

// Swap the inference request variables
void Network::swapInferenceRequest()
{
    currInfReq.swap(nextInfReq);
}
