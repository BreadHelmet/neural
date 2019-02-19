
#include "net.h"
#include <vector>
#include <cstdio>

#include <fstream>
#include <sstream>
#include <string>

#include <iostream>

#include <stdlib.h>
#include <math.h>

void runFromFile()
{
    std::ifstream infile("train-data.txt");
    std::string line;

    std::getline(infile, line);
    std::istringstream iss(line);
    
    int layer_1_topology, layer_2_topology, layer_3_topology;
    if(!(iss >> layer_1_topology >> layer_2_topology >> layer_3_topology)){
        std::cout << "Error reading topology from file. Shutting down." << std::endl;
        return;
    }

    std::vector<unsigned> topology;
    topology.push_back(layer_1_topology);
    topology.push_back(layer_2_topology);
    topology.push_back(layer_3_topology);
    Net myNet(topology);

    std::vector<double> inputValues;
    std::vector<double> resultValues;
    std::vector<double> targetValues;

    // std::cout << "beginning filestream...\n";
    while(std::getline(infile, line))
    {
            
        // std::cout << "nr_train: " << nr_train << " ";
        std::istringstream iss(line);
        double input1, input2, shouldOutput;
        if(!(iss >> input1 >> input2 >> shouldOutput)){
            std::cout << "\nreading input from file failed...\n" << std::endl;
            break;
        }
        
        inputValues.push_back(input1);
        inputValues.push_back(input2);
        targetValues.push_back(shouldOutput);

        myNet.feedForward(inputValues);
        myNet.getResults(resultValues);
        myNet.backProp(targetValues);

        std::cout << input1 << ", " << input2 << " -> ";
        for(unsigned resultIndex=0, resultsSize=resultValues.size();resultIndex<resultsSize;++resultIndex)
        {
            std::cout << round(resultValues[resultIndex]) << " ";
        }
        std::cout << std::endl;
        
        // std::cout << "Target: " << shouldOutput << std::endl;
        
        inputValues.clear();
        targetValues.clear();

        std::cout << std::string(40, '=') << std::endl;
    }
}

int main()
{
    srand(time(0));
    runFromFile();
    return 0;
}
