#ifndef NEURON
#define NEURON

#include <vector>
//class Neuron;
// typedef std::vector<Neuron> Layer;

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron
{
    typedef std::vector<Neuron> Layer;
private:
    static double eta; // overall net training rate
    static double alpha; // multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(const Layer &prevLayer);
    void setOutputVal(double val);
    double getOutputVal(void) const;
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
};

#endif // NEURON
