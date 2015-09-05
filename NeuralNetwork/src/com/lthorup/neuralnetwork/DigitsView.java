package com.lthorup.neuralnetwork;

import java.awt.Color;
import java.awt.EventQueue;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URL;
import java.util.Timer;
import java.util.TimerTask;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JTextField;
import javax.swing.border.BevelBorder;
import javax.swing.filechooser.FileNameExtensionFilter;

public class DigitsView extends JPanel {
	private JProgressBar progressBar;
	private JProgressBar accuracyBar;
	private JTextField digit;
	
	private NeuralNetwork nn;
	private ImageSet testCases;
	private ImageSet trainingSamples;
	
	private int currImage = 0;
	private ImageView imageView;
	private JTextField txtStatus;

	/**
	 * Create the panel.
	 */
	public DigitsView() {
		setLayout(null);
		
		JButton btnTrainNetwork = new JButton("Train Network");
		btnTrainNetwork.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				int sessionCnt = 20;
				int batchSize = 10;
				double learningRate = 3;
				int testCaseCnt = 100;
				nn.startTraining(testCases.getSamples(), sessionCnt, batchSize, learningRate, testCases.getSamples(), testCaseCnt);
			}
		});
		btnTrainNetwork.setBounds(89, 168, 117, 29);
		add(btnTrainNetwork);
		
		progressBar = new JProgressBar();
		progressBar.setBounds(101, 209, 217, 20);
		add(progressBar);
		
		accuracyBar = new JProgressBar();
		accuracyBar.setBounds(101, 241, 217, 20);
		add(accuracyBar);
		
		JLabel lblNewLabel = new JLabel("Progress");
		lblNewLabel.setBounds(29, 209, 61, 16);
		add(lblNewLabel);
		
		JLabel lblAccuracy = new JLabel("Accuracy");
		lblAccuracy.setBounds(29, 241, 61, 16);
		add(lblAccuracy);
		
		JButton btnLookup = new JButton("Lookup");
		btnLookup.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				BufferedImage img = new BufferedImage(imageView.getWidth(), imageView.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
				imageView.print(img.getGraphics());
				byte[] imageBytes = ((DataBufferByte) img.getData().getDataBuffer()).getData();
				int size = imageBytes.length;
				
				double[][] data = new double[112][112];
				int n = 0;
				for (int y = 0; y < 112; y++)
					for (int x = 0; x < 112; x++)
						data[x][y] = (int)(imageBytes[n++] & 0xFF) / 255.0;
				
				int inputSize = 28*28;
				double[] inputs = new double[inputSize];
				n = 0;
				for (int y = 0; y < 28; y++) {
					int sy = y*4;
					for (int x = 0; x < 28; x++) {
						int sx = x*4;
						double sum = 0.0;
						for (int dy = 0; dy < 4; dy++)
							for (int dx = 0; dx < 4; dx++)
								sum += data[sx+dx][sy+dy];
						inputs[n++] = sum / 16.0;
					}
				}
				double[] outputs = nn.evaluate(inputs);
				int id = Image.idFromVector(outputs, 10);
				digit.setText(String.valueOf(id));
				
				Graphics g = imageView.getGraphics();
				n = 0;
				for (int y = 0; y < 28; y++)
					for (int x = 0; x < 28; x++)
					{
						int v = (int)(inputs[n++] * 255);
						g.setColor(new Color(v,v,v));
						g.fillRect(x*4, y*4, 4, 4);
					}
			}
		});
		btnLookup.setBounds(177, 70, 117, 29);
		add(btnLookup);
		
		digit = new JTextField();
		digit.setBounds(239, 30, 39, 28);
		add(digit);
		digit.setColumns(10);
		
		JLabel lblDigit = new JLabel("Digit");
		lblDigit.setBounds(188, 36, 39, 16);
		add(lblDigit);
		
		JButton btnStopTraining = new JButton("Stop Training");
		btnStopTraining.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				nn.stopTraining();				
			}
		});
		btnStopTraining.setBounds(201, 168, 117, 29);
		add(btnStopTraining);
		
		JButton btnClear = new JButton("Clear");
		btnClear.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				imageView.clear();
			}
		});
		btnClear.setBounds(177, 111, 117, 29);
		add(btnClear);
		
		imageView = new ImageView();
		imageView.setBorder(new BevelBorder(BevelBorder.LOWERED, null, null, null, null));
		imageView.setBackground(Color.BLACK);
		imageView.setBounds(30, 30, 112, 112);
		add(imageView);
		
		JButton btnSave = new JButton("Save");
		btnSave.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				JFileChooser fileChooser = new JFileChooser();
				fileChooser.setFileFilter(new FileNameExtensionFilter("Maze Files", new String[] {"mz"}));
				if (fileChooser.showSaveDialog(imageView) == JFileChooser.APPROVE_OPTION) {
				    File file = fileChooser.getSelectedFile();
				    saveNetwork(file);
				}
			}
		});
		btnSave.setBounds(166, 273, 74, 29);
		add(btnSave);
		
		JButton btnLoad = new JButton("Load");
		btnLoad.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				JFileChooser fileChooser = new JFileChooser();
				fileChooser.setFileFilter(new FileNameExtensionFilter("Neural Network Files", new String[] {"nn"}));
				if (fileChooser.showOpenDialog(imageView) == JFileChooser.APPROVE_OPTION) {
				    File file = fileChooser.getSelectedFile();
				    loadNetwork(file);
				}	
			}
		});
		btnLoad.setBounds(237, 271, 81, 29);
		add(btnLoad);
		
		txtStatus = new JTextField();
		txtStatus.setBounds(73, 273, 92, 28);
		add(txtStatus);
		txtStatus.setColumns(10);
		
		JLabel lblStatus = new JLabel("Status:");
		lblStatus.setBounds(23, 278, 49, 16);
		add(lblStatus);

		// load default network (pre-trained)
		URL url = getClass().getClassLoader().getResource("resources/default.nn");
		File file = new File(url.getPath());
		loadNetwork(file);
		//nn = new NeuralNetwork(784, 100, 10);
		
		testCases = new ImageSet("test", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
		//trainingSamples = new ImageSet("samples", "train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		
        Timer t = new Timer();
        t.scheduleAtFixedRate(new TimerTask() { public void run() { update(); } }, 0, 1000);
	}
	
	private void saveNetwork(File file) {
		FileOutputStream f;
		ObjectOutputStream out;
		try {
			f = new FileOutputStream(file);
			out = new ObjectOutputStream(f);
			out.writeObject(nn);
		}
		catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	private void loadNetwork(File file) {
		FileInputStream f;
		ObjectInputStream in;
		try {
			f = new FileInputStream(file);
			in = new ObjectInputStream(f);
			nn = (NeuralNetwork) in.readObject();
			//repaint();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void update() {
		progressBar.setValue(nn.progress());
		accuracyBar.setValue(nn.accuracy());
		txtStatus.setText(nn.training() ? "training..." : "done");
	}
}
