package com.lthorup.neuralnetwork;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
import java.util.ArrayList;

import javax.swing.JPanel;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;

public class ImageView extends JPanel {

	private final int SIZE = 10;
	private final int HSIZE = SIZE/2;
	private ArrayList<Point> points = new ArrayList<Point>();
	
	public ImageView() {
		addMouseMotionListener(new MouseMotionAdapter() {
			@Override
			public void mouseDragged(MouseEvent e) {
				points.add(new Point(e.getX()-HSIZE, e.getY()-HSIZE));
				repaint();
			}
		});
		addMouseListener(new MouseAdapter() {
			@Override
			public void mousePressed(MouseEvent e) {
				points.add(new Point(e.getX()-HSIZE, e.getY()-HSIZE));
				repaint();
			}
		});
	}
	
	public void clear() {
		points.clear();
		repaint();
	}
	
	@Override
	public void paint(Graphics g) {
		g.setColor(Color.black);
		g.fillRect(0, 0, getWidth(), getHeight());
		g.setColor(Color.white);
		for (Point p : points) {
			g.fillOval(p.x, p.y, SIZE, SIZE);
		}
	}

}
