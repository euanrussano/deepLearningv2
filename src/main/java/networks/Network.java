package networks;

import java.util.List;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;

import datasets.Dataset;
import datasets.Item;
import layers.Layer;
import losses.Loss;
import losses.Metric;
import optimizers.Optimizer;

public class Network {
	
	List<Layer> layers;
	Optimizer optimizer;
	Loss loss;
	Metric metric;
	
	public void add(Layer layer) {
		layers.add(layer);
	}
	
	public void compile(Optimizer optimizer, Loss loss, Metric metric) {
		this.optimizer = optimizer;
		this.loss = loss;
		this.metric = metric;
		
		// connect dimensions of layers
		int numUnits = layers.get(0).getNumUnits(); 
		for (int i=0; i< layers.size(); i++) {
			if (i != 0) {
				layers.get(i).configure(numUnits);
			}
		}
	}
	
	public RealMatrix predict(Dataset dataset) {
		int numRows = dataset.getLength();
		
		RealMatrix yPred = MatrixUtils.createRealMatrix(new double[numRows][1]);
		
		Item item;
		RealMatrix yrow;
		
		while (dataset.hasNext()) {
			int rowNum = dataset.getCurrentRow();
			item = dataset.next();
			yrow = forward(item.getFeatures());
			yPred.setRowMatrix(rowNum, yrow);
		}
		
		return yPred;
	}
	private RealMatrix forward(RealMatrix X) {
		
		RealMatrix a = X.copy();
		
		for (Layer layer : layers) {
			a = layer.forward(X);
		}
		
		return a;
		
	}
	

}
