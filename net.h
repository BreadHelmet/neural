#ifndef NEURAL_NET
#define NEURAL_NET
#include "neuron.h"

#include <vector>
class Neuron;
typedef std::vector<Neuron> Layer;

class Net
{
private:
    std::vector<Layer> m_layers; // m_layers[layer_num][neuron_num]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError();
};

#endif // NEURAL_NET
