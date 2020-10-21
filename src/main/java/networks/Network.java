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
import losses.Loss;
import losses.Metric;
import optimizers.Optimizer;

public class Network {
	
	List<Layer> layers = new ArrayList<Layer> ();
	Optimizer optimizer;
	Loss loss;
	Loss metric;
	
	public void add(Layer layer) {
		layers.add(layer);
	}
	
	public void compile(Optimizer optimizer, Loss loss, Loss metric) {
		this.optimizer = optimizer;
		this.loss = loss;
		this.metric = metric;
		
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
	
	public static Network createFromJSON(String path) {
		// https://stackoverflow.com/questions/2591098/how-to-parse-json-in-java
		/*
		 * {
			  
			  "layers": [
			    {
			      "type": "Dense",
			      "numUnits": 1,
			      "activation": "relu",
			      "input_shape": 2,
			      "weights": {{0.87}, {-2.3}}
			      "bias": {{0.0}}
			    },
			    {
			      "type": "Dense",
			      "numUnits": 2,
			      "activation": "sigmoid",
			      "input_shape": 10,
			      "weights": {{1.74, -0.76}};
			      "bias": {{0.0, 0.0}};
			    }
			  ],
			  "metrics": 
			  {
			  	"name":"accuracy",
			  }
			  "optimizer": {
			  	"name": "sgd",
			  	"learningRate": 0.01
			  	}
			}
		 */
		String content;
		JsonObject object = null;
		try {
			content = new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8);
			object = Json.parse(content).asObject();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		String optimizer = ((JsonObject) object.get("optimizer")).get("name").toString();
		double learningRate= ((JsonObject) object.get("optimizer")).get("learningRate").asDouble();
		
		String metrics = object.get("metrics").toString();
		
		JsonArray layersJson= object.get("layers").asArray();
		JsonObject layerJson;
		for (JsonValue obj : layersJson) {
			layerJson = (JsonObject) obj;
		}
		
		return null;
	}
	
	public boolean saveAsJson() throws IOException { 
		
		JsonObject obj = Json.object()
						.add("metric", metric.toJson())
						.add("optimizer", optimizer.toJson());
		
		JsonArray jsonLayers = new JsonArray();
		for (Layer layer : layers) {
			jsonLayers.add(layer.toJson());
		 }
		
		obj.add("layers",jsonLayers);
		
		
		BufferedWriter writer = new BufferedWriter(new FileWriter("model.json"));
	    writer.write(obj.toString());
	    
	    writer.close();
	    
		
		return true;
	}
	

}
