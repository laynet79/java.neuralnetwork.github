package com.lthorup.neuralnetwork;

public interface NeuralSample {
	public double[] input();
	public double[] output();
	public boolean matches(double[] selection);
}
