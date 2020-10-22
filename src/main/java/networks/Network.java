package networks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;

import datasets.Dataset;
import datasets.Item;
import layers.Layer;
import layers.LayerFactory;
import losses.Loss;
import losses.Metric;
import optimizers.Optimizer;

public class Network {
	
	private List<Layer> layers = new ArrayList<Layer> ();
	private Optimizer optimizer;
	private Loss loss;
	
	public void add(Layer layer) {
		layers.add(layer);
	}
	
	public void compile(Optimizer optimizer, Loss loss) {
		this.optimizer = optimizer;
		this.loss = loss;
		
		// connect dimensions of layers
		int inputShape = layers.get(0).getInputShape();
		
		for (int i=0; i< layers.size(); i++) {
			layers.get(i).configure(inputShape);
			inputShape = layers.get(i).getNumUnits();
		}
		
		// save the layers in the optimizer
		this.optimizer.configure(layers);
	}
	
	public RealMatrix predict(Dataset dataset) {
		int numRows = dataset.getLength();
		int numCols = dataset.getTargetDimension();
		
		RealMatrix yPred = MatrixUtils.createRealMatrix(new double[numRows][numCols]);
		
		Item batch;
		RealMatrix batch_pred;
		
		dataset.reset();
		while (dataset.hasNext()) {
			int currentRow = dataset.getCurrentRow();
			batch = dataset.next();
			batch_pred = forward(batch.getFeatures());
			yPred.setSubMatrix(batch_pred.getData(), currentRow , 0);
		}
		
		return yPred;
	}
	private RealMatrix forward(RealMatrix X) {
		
		RealMatrix a = X.copy();
		
		for (Layer layer : layers) {
			//System.out.println("a_in = " + a);
			//System.out.println("w = " + layer.getWeights());
			a = layer.forward(a);
			//System.out.println("a_out = " + a);
		}
		
		return a;
		
	}
	
	private void backward(RealMatrix loss_grad) {
		
		RealMatrix deltaout = loss_grad.copy();
		
		for (int i = layers.size()-1; i>=0; i--) {
			deltaout = layers.get(i).backward(deltaout);
		}
	}
	
	public void fit(Dataset dataset, double epochs) {
	
		Item batch;
		RealMatrix batch_pred;
		
		for (int i=0; i<epochs; i++) {
			dataset.reset();
			while (dataset.hasNext()) {
				//System.out.println("w2 = " + layers.get(0).getWeights());
				//System.out.println("dw2 = " + layers.get(0).getDeltaWeights());
				batch = dataset.next();
				
				// forward pass
				batch_pred = forward(batch.getFeatures());
				//System.out.println("batch_pred  = " + batch_pred );
				// calculate loss and der
				double lossValue = loss.evaluateSum(batch.getTarget(), batch_pred);		
				RealMatrix loss_grad = loss.evaluateDer(batch.getTarget(), batch_pred);
				//System.out.println("loss_grad = " + loss_grad);
				System.out.println("epoch " + i + "     loss = " + lossValue);
				// backward pass
				backward(loss_grad);
				
				// optimizer update weights
				optimizer.update();
				
			}
		}

	}

	public List<Layer> getLayers() {
		return layers;
	}

	public void setLayers(List<Layer> layers) {
		this.layers = layers;
	}

	public Optimizer getOptimizer() {
		return optimizer;
	}

	public void setOptimizer(Optimizer optimizer) {
		this.optimizer = optimizer;
	}

	public Loss getLoss() {
		return loss;
	}

	public void setLoss(Loss loss) {
		this.loss = loss;
	}
	
	public static boolean saveAsJson(Network model) throws IOException { 
		
		JsonObject obj = Json.object()
						.add("optimizer", model.getOptimizer().toJson())
						.add("loss", model.getLoss().toJson());
		
		JsonArray jsonLayers = new JsonArray();
		for (Layer layer : model.getLayers()) {
			jsonLayers.add(layer.toJson());
		 }
		
		obj.add("layers",jsonLayers);
		
		
		BufferedWriter writer = new BufferedWriter(new FileWriter("model.json"));
	    writer.write(obj.toString());
	    
	    writer.close();
	    
		
		return true;
	}
	
	@Override
	public String toString() {
		String out = "Network with " + this.getLayers().size() + " layers.\n" +
					 "Optimizer = " + this.getOptimizer().toString() + "\n" +
					 "Loss = " + this.getLoss().toString() + "\n";
		
		out += 	"--".repeat(20) + "\n" +
				"Layers" + "\n" +
				"--".repeat(20) + "\n";
		
		for (Layer layer : layers) {
			out += layer.toString() + "\n";
		}
		
		out += 	"--".repeat(20) + "\n";
		
		return out;
		
	}
	

}

