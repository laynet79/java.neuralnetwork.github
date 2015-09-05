package com.lthorup.neuralnetwork;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.net.URL;

/*
 * This Class is designed to read in 
 */
public class ImageSet implements NeuralSampleSet {
	
	private final int CNT_OFFSET = 4;
	private final int RESY_OFFSET = 8;
	private final int RESX_OFFSET = 12;
	private final int IMAGE_OFFSET = 16;
	private final int LABEL_OFFSET = 8;
	
	private String name;
	private int imageSize;
	private float[] imageData;
	private NeuralSample[] images;
		
	//public static void main(String[] args) {
	//	ImageSet data = new ImageSet("test", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	//}
	
	public ImageSet(String name, String imageFile, String labelFile) {
		
		this.name = name;
		int cnt = 0;
		
		try {
			// read in image file data
			URL url = getClass().getClassLoader().getResource("resources/" + imageFile);
			File file = new File(url.getPath());
			int fileSize = (int) file.length();
			byte[] imageData = new byte[(int) file.length()];
			DataInputStream d = new DataInputStream(new FileInputStream(file));
			d.readFully(imageData);
			d.close();
			
			int imageCnt = parseInt(imageData, CNT_OFFSET);
			int resY = parseInt(imageData, RESY_OFFSET);
			int resX = parseInt(imageData, RESX_OFFSET);
			imageSize = resX * resY;
			images = new Image[imageCnt];
			
			// read in label data file
			url = getClass().getClassLoader().getResource("resources/" + labelFile);
			file = new File(url.getPath());
			byte[] labelData = new byte[(int) file.length()];
			int len = labelData.length;
			d = new DataInputStream(new FileInputStream(file));
			d.readFully(labelData);
			d.close();
			
			int labelCnt = parseInt(labelData, CNT_OFFSET);
			if (labelCnt != imageCnt)
				throw new Exception("image file and label file count missmatch");

			// create list if images from file data
			imageSize = resX * resY;
			images = new Image[imageCnt];
			for (int i = 0; i < imageCnt; i++) {
				if (i == 9992)
					cnt = cnt;
				int label = labelData[LABEL_OFFSET+i];
				images[i] = new Image(label, 10, imageData, IMAGE_OFFSET + imageSize*i, imageSize);
				cnt++;
			}
			System.out.println("operation complete");
		}
		catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	public String name() { return name; }
	public NeuralSample[] getSamples() { return images; }
	public int inputSize() { return imageSize; }
	public int outputSize() { return 10; }
	
	private int parseInt(byte[] data, int offset) {
		int value = 0;
		for (int i = 0; i < 4; i++)
			value = (value << 8) + data[offset+i];
		return value;
	}
}
