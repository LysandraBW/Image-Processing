#pragma once

#include <iostream>
#include "Node.h"

using namespace std;

struct Layer {
	int numberNodes = 0;
	Node* nodes = nullptr;

	Layer* prevLayer = nullptr;
	Layer* nextLayer = nullptr;

	Layer(int& numberNodes, Layer* nextLayer) {
		this->numberNodes = numberNodes;
		this->nodes = new Node[numberNodes];

		this->nextLayer = nextLayer;
		
		if (nextLayer == nullptr)
			return;

		for (int n = 0; n < numberNodes; n++)
			this->nodes[n].setNodeWeightNumber(nextLayer->numberNodes);
	}

	void forwardPass(Layer* inputLayer) {
		//cout << "Forward Pass From Layer " << inputLayer << endl;
		for (int n = 0; n < this->numberNodes; n++) {
			nodes[n].calculateValue(n, inputLayer->nodes, inputLayer->numberNodes);
			//cout << "Node " << n + 1 << " Calculated Value: " << nodes[n].value << endl << endl;
		}

		if (nextLayer != nullptr)
			nextLayer->forwardPass(this);
	}

	void forwardPass(double* initialInput) {
		//cout << "Initial Forward Pass w Layer " << this << endl;
		for (int n = 0; n < this->numberNodes; n++) {
			nodes[n].value = initialInput[n];
			// cout << nodes[n].value << endl;
		}

		if (nextLayer != nullptr)
			nextLayer->forwardPass(this);
	}

	void calculateDelta(double* outputData = nullptr) {
		// Output Layer
		if (this->nextLayer == nullptr) {
			for (int n = 0; n < this->numberNodes; n++) {
				this->nodes[n].delta = 2 * (this->nodes[n].value - outputData[n]) * (this->nodes[n].value * (1 - this->nodes[n].value));
			}
			this->prevLayer->calculateDelta();
			return;
		}

		// Hidden Layer
		for (int n = 0; n < this->numberNodes; n++) {
			this->nodes[n].delta = 0.0;
			for (int w = 0; w < this->nodes[n].numberWeights; w++) {
				this->nodes[n].delta += this->nextLayer->nodes[w].delta * this->nodes[n].weights[w] * (this->nodes[n].value * (1.0 - this->nodes[n].value));
			}
		}

		if (this->prevLayer != nullptr)
			this->prevLayer->calculateDelta();
	}

	void backwardPass(double learningRate) {
		for (int n = 0; n < this->numberNodes; n++) {
			this->nodes[n].bias -= learningRate * this->nodes[n].delta;
			for (int w = 0; w < this->nodes[n].numberWeights; w++) {
				this->nodes[n].weights[w] -= learningRate * this->nextLayer->nodes[w].delta * this->nodes[n].value;
			}
		}

		if (this->prevLayer != nullptr)
			this->prevLayer->backwardPass(learningRate);
	}

	void breakdown() {
		cout << "Layer (" << this << ") Breakdown:" << endl;
		cout << "\tNumber of Nodes: " << this->numberNodes << endl;
		for (int n = 0; n < this->numberNodes; n++) {
			cout << "\t\tNode " << n << ": " << endl;
			cout << "\t\t\tValue: " << this->nodes[n].value << endl;
			cout << "\t\t\tBias: " << this->nodes[n].bias << endl;
			cout << "\t\t\tWeights:" << endl;
			for (int w = 0; w < this->nodes[n].numberWeights; w++) {
				cout << "\t\t\t\t" << w << ": " << this->nodes[n].weights[w] << endl;
			}
			cout << "\t\t\tDelta: " << this->nodes[n].delta << endl;
		}
	}
};