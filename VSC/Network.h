#pragma once

#include <vector>
#include <cmath>
#include "Layer.h"

using namespace std;

struct Network {

	vector<int> nodesPerLayer;
	vector<Layer*> layers;
	
	Network(vector<int>& nodesPerLayer) {
		this->nodesPerLayer = nodesPerLayer;
		this->layers.resize(nodesPerLayer.size());

		for (int l = nodesPerLayer.size() - 1; l >= 0; l--) {
			this->layers[l] = new Layer(nodesPerLayer[l], ((l == nodesPerLayer.size() - 1) ? nullptr : this->layers[l + 1]));
		}

		for (int l = 0; l < nodesPerLayer.size(); l++) {
			this->layers[l]->prevLayer = (l == 0 ? nullptr : this->layers[l - 1]);
		}
	}

	void trainNetwork(double** trainingData, int trainingDataRows, double* (extractInput(double* data)), double* (extractOutput(double* data)), int epochs = 10, double learningRate = 0.1, int batchSize = 10) {
		int epoch = 0;

		while (epoch++ < epochs) {
			// cout << "Before: \n";
			// for (int l = 0; l < this->layers.size(); l++)
			// 	this->layers[l]->breakdown();
			for (int t = 0; t < trainingDataRows; t++) {
				double* inputData = extractInput(trainingData[t]);
				double* outputData = extractOutput(trainingData[t]);
				this->layers[0]->forwardPass(inputData);
				this->layers[this->layers.size() - 1]->calculateDelta(outputData);
				this->layers[this->layers.size() - 1]->backwardPass(learningRate);
			}
			// cout << "After: \n";
			// for (int l = 0; l < this->layers.size(); l++)
			// 	this->layers[l]->breakdown();
		}
	}

	void testNetwork(double** testData, int testDataRows, double* (extractInput(double* data)), double* (extractOutput(double* data))) {
		int wrongOutputs = 0;
		bool c = true;
		for (int t = 0; t < testDataRows; t++) {
			c = true;
			//cout << "Test Batch #" << t + 1 << endl;
			layers[0]->forwardPass(extractInput(testData[t]));
			double* correctOutput = extractOutput(testData[t]);

			double maxNum = 0.0;
			int maxIndex = 0;

			int cOIndex = 0;

			for (int n = 0; n < this->layers[this->layers.size() - 1]->numberNodes; n++) {
				// cout << "Node " << n + 1 << ": " << this->layers[this->layers.size() - 1]->nodes[n].value << endl;
				// cout << "\tExpected Output: " << correctOutput[n] << endl;
				// if (c && abs(this->layers[this->layers.size() - 1]->nodes[n].value - correctOutput[n]) > 0.1) {
				// 	wrongOutputs++;
				// 	c = false;
				// }

				if (this->layers[this->layers.size() - 1]->nodes[n].value > maxNum) {
					maxNum = this->layers[this->layers.size() - 1]->nodes[n].value;
					maxIndex = n;
				}

				if ((int) correctOutput[n] == 1)
					cOIndex = n;
				
			}

			if (maxIndex != cOIndex) {
				wrongOutputs++;
			}

			cout << endl;
		}

		cout << "Accuracy: " << testDataRows - wrongOutputs << "/" << testDataRows << endl;
	}

	void setNodeParameters(double weight, double bias) {
		for (int l = 0; l < layers.size(); l++) {
			int numberNodes = layers[l]->numberNodes;
			for (int n = 0; n < numberNodes; n++) {
				layers[l]->nodes[n].setParameters(weight, bias);
			}
		}
	}

	void setNodeRandomParameters(double minWeight, double maxWeight, double minBias, double maxBias) {
		for (int l = 0; l < layers.size(); l++) {
			int numberNodes = layers[l]->numberNodes;
			for (int n = 0; n < numberNodes; n++) {
				layers[l]->nodes[n].setParameters(0.0, 0.0, true, minWeight, maxWeight, minBias, maxBias);
			}
		}
	}
};