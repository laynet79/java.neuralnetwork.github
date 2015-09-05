package com.lthorup.neuralnetwork;

class Image implements NeuralSample {
	private int id;
	private int numIds;
	private double[] idVector;
	private double[] data;
	private int dataSize;
	
	public Image(int id, int numIds, byte[] data, int dataOffset, int dataSize) {
		this.id = id;
		this.numIds = numIds;
		idVector = new double[numIds];
		for (int i = 0; i < numIds; i++)
			idVector[i] = 0.0;
		idVector[id] = 1.0;
		this.dataSize = dataSize;
		this.data = new double[dataSize];
		for (int i = 0; i < dataSize; i++)
			this.data[i] = (short)(data[i+dataOffset] & 0xFF) / 255.0;
	}
	public int id() { return id; }
	public int dataSize()  { return dataSize;  }
	public double[] data() { return data; }

	public double[] output() { return idVector; }
	public double[] input()  { return data; }
	
	public boolean matches(double[] selection) {
		int max = 0;
		for (int i = 0; i < numIds; i++)
			if (selection[i] > selection[max])
				max = i;
		return max == id;
	}
	
	public static int idFromVector(double[] idVector, int numIds) {
		int max = 0;
		for (int i = 0; i < numIds; i++)
			if (idVector[i] > idVector[max])
				max = i;
		return max;
	}
}
