package optimizers;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonObject;

import layers.Dense;
import layers.Layer;

public abstract class Optimizer {
	
	List<Layer> layers = new ArrayList<Layer>();
	double learningRate;
	String name;
	
	public Optimizer(double learning_rate) {
		this.learningRate = learning_rate;
		this.name = "";
	}
	
	// store the layers in the optimizer for the update process
	public void configure(List<Layer> layers) {
		this.layers = layers;
	}
	
	public void update() {
		
		RealMatrix weights;
		RealMatrix gradWeights;
		RealMatrix bias;
		RealMatrix gradBias;
		
		for (Layer layer : layers) {
			if (layer.getClass().equals(Dense.class)) {
				weights = layer.getWeights();
				gradWeights = layer.getDeltaWeights();
				
				bias = layer.getBias();
				gradBias = layer.getDeltaBias();
				
				weights = updateSingle(weights, gradWeights);
				bias = updateSingle(bias, gradBias);
				
				layer.setBias(bias);
				layer.setWeigths(weights);
			}
		}
		
	}

	protected abstract RealMatrix updateSingle(RealMatrix weights, RealMatrix gradWeights);

	public JsonObject toJson() {
		JsonObject obj = Json.object()
				.add("name", name)
				.add("learningRate", learningRate);
		
		return obj;
	}
}
