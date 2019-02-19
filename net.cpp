
#include "net.h"
#include <cassert>
#include "math.h"
#include <iostream>

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for(unsigned layerNum=0;layerNum<numLayers;++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for(unsigned neuronNum=0;neuronNum<=topology[layerNum];++neuronNum)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            // std::cout << "Made a neuron!" << std::endl;
        }

        // Force the bias node's output value to 1.0. It's the last neuron created above
        m_layers.back().back().setOutputVal(1.0);
    }

    m_recentAverageError = 0.0;
    // m_error = 0.0;
    m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over.
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size()-1);

    // input layer
    for(unsigned i=0,numInputVals=inputVals.size();i<numInputVals;++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagate
    for(unsigned layerNum=1, layersSize=m_layers.size();layerNum<layersSize;++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum -1];
        for(unsigned n=0, l;n<m_layers[layerNum].size()-1;++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors, Root Mean Square error)

    Layer &outputLayer = m_layers.back();
    m_error = 0.0; // should be a member

    unsigned layerSizeM1 = outputLayer.size()-1;
    for(unsigned n=0;n<layerSizeM1;++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    // std::cout << "m_error ( after squaring ): " << m_error << std::endl;

    m_error /= (outputLayer.size()-1);
    // std::cout << "m_error ( after dividing ): " << m_error << std::endl;
    m_error = sqrt(m_error); // RMS
    // std::cout << "m_error: " << m_error << std::endl;


    // this grows ridiculously large!
    // m_recentAverageError = (m_recentAverageError + m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1);
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);
    // std::cout << "error / average e. : " << m_error << " / " << m_recentAverageError << std::endl;
    // std::cout << "m_recentAverageError: " << m_recentAverageError << std::endl;

    // Calculate output layer gradients
    for(unsigned n=0;n<layerSizeM1;++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate hidden layer gradients
    for(unsigned layerNum = m_layers.size() - 2;layerNum > 0;--layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for(unsigned n=0;n<hiddenLayer.size();++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer.
    // update connection weights

    for(unsigned layerNum = m_layers.size()-1;layerNum > 0;--layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum-1];

        for(unsigned n = 0;n<layer.size()-1;++n)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &resultVals) const
{
    resultVals.clear();

    for(unsigned n=0;n<m_layers.back().size()-1;++n)
    {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

double Net::getRecentAverageError()
{
    return m_recentAverageError;
}
