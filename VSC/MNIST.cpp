#include <iostream>
#include <fstream>
#include <string>
#include "Network.h"
using namespace std;

void separateLineCSV(double* data, const int& size, const string& line) {
    int start = 0;
    int length = 0;
    int index = 0;

    for (int l = 0; l < line.length(); l++) {
        if (line[l] != ',')
            continue;
        length = l - start;
        data[index] = (double) stoi(line.substr(start, length));

        start = l + 1;
        index++;
    }

    data[index] = (double) stoi(line.substr(start, line.length() - start));
}

void loadDataCSV(double** data, int rows, int columns, ifstream& is) {
    string line;
    int index = 0;

    getline(is, line);

    while (getline(is, line) && index < rows) {
        separateLineCSV(data[index], columns, line);
        index++;
    }
}

double* extractInputs(double* data) {
    double* inputs = new double[784];
    // cout << "Inputs" << endl;
    for (int i = 1; i < 785; i++) {
        inputs[i] = data[i];
    }
    for (int i = 1; i < 785; i++) {
        // cout << inputs[i] << " ";
    }
    // cout << endl;
    return inputs;
}

double* extractOutputs(double* data) {
    double* output = new double[10];
    
    int onDigit = (int) data[0];

    // cout << "Outputs" << endl;
    for (int i = 0; i < 10; i++) {
         output[i] = (i == onDigit) ? 1.0 : 0.0;
    }
    for (int i = 0; i < 10; i++) {
        // cout << output[i] << " ";
    }
    // cout << endl;

    return output;
}

int main() {
    ifstream trainingData("./mnist_train.csv");

    if (!trainingData)
       return -454;

    double** data = new double* [60000];
    for (int d = 0; d < 60000; d++)
       data[d] = new double[785];

    loadDataCSV(data, 60000, 785, trainingData);
    trainingData.close();

    ifstream testingData("./mnist_test.csv");

    if (!testingData)
        return -545;

    double** testData = new double* [10000];
    for (int d = 0; d < 10000; d++)
        testData[d] = new double[785];

    loadDataCSV(testData, 10000, 785, testingData);
    testingData.close();

    vector<int> layerBreakdown = { 784, 16, 16, 10 };
    Network network(layerBreakdown);
    network.setNodeRandomParameters(-1.0, 1.0, -1.0, 1.0);

    network.testNetwork(data, 60000, &extractInputs, &extractOutputs);
    network.trainNetwork(data, 60000, &extractInputs, &extractOutputs, 22);
    network.testNetwork(data, 60000, &extractInputs, &extractOutputs);
}
