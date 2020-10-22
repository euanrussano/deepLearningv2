package layers;

import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;

import functions.ActivationFunction;

public abstract class Layer {
	
	private RealMatrix weights;
	private RealMatrix bias;
	private ActivationFunction func;
	private int inputShape;
	private int numUnits;
	private RealMatrix input;
	private RealMatrix netSum;
	private RealMatrix deltaW;
	private RealMatrix deltab;
	private String type;
	
	boolean configured;
	
	
	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public void setNumUnits(int numUnits) {
		this.numUnits = numUnits;
	}

	public ActivationFunction getFunc() {
		return func;
	}

	public void setFunc(ActivationFunction func) {
		this.func = func;
	}

	
	public boolean isConfigured() {
		return configured;
	}

	public void setConfigured(boolean configured) {
		this.configured = configured;
	}

	public void configure(int input_shape) {
		
	}
	
	public RealMatrix forward(RealMatrix X) {
		return X.copy();
	}
	
	public RealMatrix backward(RealMatrix accum_grad) {
		return null;
	}
	
	public int getNumUnits() {
		return numUnits;
	}
	
	public RealMatrix getWeights() {
		return weights;
	}
	
	public RealMatrix getBias() {
		return bias;
	}
	
	public RealMatrix getDeltaWeights() {
		return this.deltaW;
	}
	
	public RealMatrix getDeltaBias() {
		return this.deltab;
	}
	
	public void setDeltaWeights(RealMatrix deltaWeights) {
		this.deltaW = deltaWeights;
	}
	
	public void setDeltaBias(RealMatrix deltaB) {
		this.deltab = deltaB;
	}
	
	public void setBias(RealMatrix bias) {
		this.bias = bias;
	}
	
	public void setWeigths(RealMatrix weights) {
		this.weights = weights;
	}
	
	public int getInputShape() {
		return inputShape;
	}

	public void setInputShape(int inputShape) {
		this.inputShape = inputShape;
	}

	public JsonObject toJson() {
		/*
		 * {
			      "type": "Dense",
			      "numUnits": 1,
			      "activation": "relu",
			      "input_shape": 2,
			      "weights": {{0.87}, {-2.3}}
			      "bias": {{0.0}}
			    },
		 */
		JsonArray weightsJson = new JsonArray();
		JsonArray row;
		 for (int i=0; i<weights.getRowDimension(); i++) {
			 row = new JsonArray();
			 for (int j=0; j<weights.getColumnDimension(); j++) {
				  row.add(weights.getEntry(i, j));
			 }
			 weightsJson.add(row);
		 }
		 
		 JsonArray biasJson = new JsonArray();
			
		 for (int i=0; i<bias.getRowDimension(); i++) {
			 row = new JsonArray();
			 for (int j=0; j<bias.getColumnDimension(); j++) {
				  row.add(bias.getEntry(i, j));
			 }
			 biasJson.add(row);
		 }
		 
		JsonObject obj = Json.object()
				.add("type", type)
				.add("numUnits", numUnits)
				.add("activation", func.toJson())
				.add("inputShape", inputShape)
				.add("weights", weightsJson)
				.add("bias", biasJson);
		
		
		return obj;
		
	}

	public RealMatrix getInput() {
		return input;
	}

	public void setInput(RealMatrix input) {
		this.input = input;
	}

	public RealMatrix getNetSum() {
		return netSum;
	}

	public void setNetSum(RealMatrix netSum) {
		this.netSum = netSum;
	}

}
