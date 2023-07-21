#pragma once
#include <iostream>
#include <cmath>
using namespace std;
struct Node {
	double value = 0.0;
	double bias = 0.0;
	int numberWeights = 0;
	double* weights = nullptr;
	double delta = 0.0;

	void setNodeWeightNumber(int& numberWeights) {
		this->numberWeights = numberWeights;
		this->weights = new double[numberWeights];
		for (int w = 0; w < numberWeights; w++)
			this->weights[w] = 0.0;
	}

	void calculateValue(int& order, Node* inputNodes, int& numberInputNodes) {
		for (int n = 0; n < numberInputNodes; n++) {
			//cout << "+= " << inputNodes[n].value << " * " << inputNodes[n].weights[order] << endl;
			this->value += inputNodes[n].value * inputNodes[n].weights[order];
		}
		//cout << "+= " << this->bias << endl;
		this->value += this->bias;
		this->value = 1.0/(1+exp(-this->value));
	}

	void setParameters(double weight = 0.0, double bias = 0.0, bool random = false, double weightMin = -1.0, double weightMax = 1.0, double biasMin = -1.0, double biasMax = 1.0) {
		auto randomDouble = [](double min, double max) {
			if (min > max)
				return 0.0;
			return (rand() / (double)(RAND_MAX)) * (max - min) + min;
		};

		for (int w = 0; w < this->numberWeights; w++) {
			this->weights[w] = ((random) ? randomDouble(weightMin, weightMax) : weight);
		}
		this->bias = ((random) ? randomDouble(biasMin, biasMax) : bias);
	}
};