//----------------------------------------------------
// Reference web sites
//	https://msdn.microsoft.com/en-us/magazine/jj658979.aspx
//  http://neuralnetworksanddeeplearning.com/chap1.html
//----------------------------------------------------
package com.lthorup.neuralnetwork;

import java.io.Serializable;
import java.util.Random;

public class NeuralNetwork implements Runnable, Serializable {

	static final long serialVersionUID = 1;
	
	private int numInput, numOutput, numHidden;
	private double[][] ihWeights;
	private double[] ihBiases;
	private double[] ihSums;
	private double[] ihOutputs;
	
	private double[][] hoWeights;
	private double[] hoBiases;
	private double[] hoSums;
	private double[] outputs;
	
	private double[] hGrads;
	private double[] oGrads;
	
	private double[][] ihWeightDeltas;
	private double[] ihBiasDeltas;
	private double[][] hoWeightDeltas;
	private double[] hoBiasDeltas;
	
	private volatile boolean training;
	private volatile double accuracy;
	private volatile double progress;
	
	private volatile boolean exiting;
	private NeuralSample[] samples;
	private int batchSize;
	private int sessionCnt;
	private double learningRate;
	private NeuralSample[] testCases;
	private int testCaseCnt;

	Random rand = new Random();
	
	//------------------------------------------
	public NeuralNetwork(int numInput, int numHidden, int numOutput) {
		this.numInput = numInput;
		this.numHidden = numHidden;
		this.numOutput = numOutput;
		
		ihWeights = new double[numInput][numHidden];
		ihBiases = new double[numHidden];
		ihSums = new double[numHidden];
		ihOutputs = new double[numHidden];
		
		hoWeights = new double[numHidden][numOutput];
		hoBiases = new double[numOutput];
		hoSums = new double[numOutput];
		outputs = new double[numOutput];
		
		hGrads = new double[numHidden];
		oGrads = new double[numOutput];
		
		ihWeightDeltas = new double[numInput][numHidden];
		ihBiasDeltas = new double[numHidden];
		hoWeightDeltas = new double[numHidden][numOutput];
		hoBiasDeltas = new double[numOutput];
				
		resetWeights();
		training = false;
	}
	
	//------------------------------------------
	public int inputSize()  { return numInput;  }
	public int outputSize() { return numOutput; }
	
	//------------------------------------------
	public int accuracy() { return (int)(accuracy * 100); }
	public int progress() { return (int)(progress * 100); }
	
	//------------------------------------------
	public double[] evaluate(double[] inputs) {
		// the input layer neuron values have been provided
		// process the hidden layer
		for (int h = 0; h < numHidden; h++) {
			ihSums[h] = 0.0;
			for (int i = 0; i < numInput; i++) {
				ihSums[h] += ihWeights[i][h] * inputs[i];
			}
			ihSums[h] += ihBiases[h];
			ihOutputs[h] = sigmoid(ihSums[h]);
		}
		
		// process the output layer neurons and return them
		for (int o = 0; o < numOutput; o++) {
			hoSums[o] = 0.0;
			for (int h = 0; h < numHidden; h++) {
				hoSums[o] += hoWeights[h][o] * ihOutputs[h];
			}
			hoSums[o] += hoBiases[o];
			outputs[o] = sigmoid(hoSums[o]);
		}
		return outputs;
	}

	//------------------------------------------
	private void resetWeights() {
		for (int h = 0; h < numHidden; h++) {
			for (int i = 0; i < numInput; i++) {
				ihWeights[i][h] = rand.nextGaussian();
				ihWeightDeltas[i][h] = 0.0;
			}
			ihBiases[h] = rand.nextGaussian();
			ihBiasDeltas[h] = 0.0;
		}
		for (int o = 0; o < numOutput; o++) {
			for (int h = 0; h < numHidden; h++) {
				hoWeights[h][o] = rand.nextGaussian();
				hoWeightDeltas[h][o] = 0.0;
			}
			hoBiases[o] = rand.nextGaussian();
			hoBiasDeltas[o] = 0.0;
		}
		accuracy = 0.0;
		progress = 0.0;
	}
	
	//------------------------------------------
	public void startTraining(NeuralSample[] samples, int sessionCnt, int batchSize, double learningRate, NeuralSample[] testCases, int testCaseCnt) {
		this.samples = samples;
		this.batchSize = batchSize;
		this.sessionCnt = sessionCnt;
		this.learningRate = learningRate;
		this.testCases = testCases;
		this.testCaseCnt = testCaseCnt;
		accuracy = 0.0;
		progress = 0.0;
		Thread thread = new Thread(this);
		exiting = false;
		thread.start();
	}
	
	public void stopTraining() {
		exiting = true;
	}
	
	public boolean training() { return training; }
	
	@Override
	public void run() { train(); }
	
	//------------------------------------------
	// train the network using backpropagation stochastic gradient descent.
	// (we break up the sample set into random mini-sets and for each mini-set
	// we compute the average deltas to the weights and biases, and apply those
	// deltas at the end of each mini-set.  If a test case set of samples is
	// provided, they are tested at the end of each mini-set where the
	// accuracy of the network is updated.
	public void train() {
		training = true;
		resetWeights();
		for (int session = 0; session < sessionCnt; session++) {
			NeuralSample[] randomSamples = createRandomSamples();
			int batchStart = 0;
			
			while (batchStart < samples.length) {
				int batchEnd = Math.min(batchStart + batchSize, samples.length);
				int sampleCnt = batchEnd - batchStart;
				for (int i=batchStart; i < batchEnd; i++) {
					learnSample(randomSamples[i].input(), randomSamples[i].output());
					updateWeights(sampleCnt);					
				}
				batchStart += sampleCnt;
				if (exiting)
					break;
			}
			
			if (testCases != null)
				accuracy = runTest(testCases, testCaseCnt);
			progress = (double)(session+1) / sessionCnt;
			if (exiting)
				break;	
		}
		samples = null;
		testCases = null;
		training = false;
	}
	
	//------------------------------------------
	private NeuralSample[] createRandomSamples() {
		NeuralSample[] randomSamples = samples.clone();
		int cnt = randomSamples.length;
		for (int i = 0; i < cnt; i++) {
			int a = rand.nextInt(cnt);
			int b = rand.nextInt(cnt);
			NeuralSample temp = randomSamples[a];
			randomSamples[a] = randomSamples[b];
			randomSamples[b] = temp;
		}
		return randomSamples;
	}
		
	//------------------------------------------
	private void updateWeights(int sampleCnt) {
		double scale = learningRate/sampleCnt;
		for (int h = 0; h < numHidden; h++) {
			for (int i = 0; i < numInput; i++) {
				ihWeights[i][h] += ihWeightDeltas[i][h] * scale;
				ihWeightDeltas[i][h] = 0.0;
			}
			ihBiases[h] += ihBiasDeltas[h] * scale;
			ihBiasDeltas[h] = 0.0;
		}
		for (int o = 0; o < numOutput; o++) {
			for (int h = 0; h < numHidden; h++) {
				hoWeights[h][o] += hoWeightDeltas[h][o] * scale;
				hoWeightDeltas[h][o] = 0.0;
			}
			hoBiases[o] += hoBiasDeltas[o] * scale;
			hoBiasDeltas[o] = 0.0;
		}
	}
	
	//------------------------------------------
	private void learnSample(double[] sampleInput, double[] sampleOutput) {
		// run sample input through the current network
		evaluate(sampleInput);
		
		// compute the output layer gradients
		for (int o = 0; o < numOutput; o++)
			oGrads[o] = (sampleOutput[o] - outputs[o]) * sigmoidPrim(hoSums[o]);

		// compute the hidden layer gradients
		for (int h = 0; h < numHidden; h++) {
			double sum = 0.0;
			for (int o = 0; o < numOutput; o++) {
				sum += oGrads[o] * hoWeights[h][o];
			}
			hGrads[h] = sigmoidPrim(ihSums[h]) * sum;
		}
		
		// update input to hidden layer weights and biases
		for (int h = 0; h < numHidden; h++) {
			for (int i = 0; i < numInput; i++)
				ihWeightDeltas[i][h] += hGrads[h] * sampleInput[i];
			ihBiasDeltas[h] += hGrads[h];
		}
		
		// update hidden to output layer weights and biases
		for (int o = 0; o < numOutput; o++) {
			for (int h = 0; h < numHidden; h++)
				hoWeightDeltas[h][o] += oGrads[o] * ihOutputs[h];
			hoBiasDeltas[o] += oGrads[o];
		}
	}
	
	//------------------------------------------
	private double runTest(NeuralSample[] testCases, int testCaseCnt) {
		double sum = 0.0;
		int cnt = 0;
		for (NeuralSample sample : testCases) {
			evaluate(sample.input());
			if (sample.matches(outputs))
				sum += 1;
			cnt++;
			if (cnt == testCaseCnt)
				break;
		}
		return sum / cnt;
	}

	//------------------------------------------
	private double sigmoid(double x) {
		if (x < -45.0)
			return 0.0;
		if (x > 45.0)
			return 1.0;
		return 1.0 / (1.0 + Math.exp(-x));
	}
	private double sigmoidPrim(double x) {
		return sigmoid(x)*(1.0-sigmoid(x));
	}
	//------------------------------------------
}
