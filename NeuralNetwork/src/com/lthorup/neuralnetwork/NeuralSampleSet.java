package com.lthorup.neuralnetwork;

public interface NeuralSampleSet {

	public int inputSize();
	public int outputSize();
	public NeuralSample[] getSamples();

}
