package layers;

import org.apache.commons.math4.linear.RealMatrix;

import functions.Function;

public abstract class Layer {
	
	RealMatrix weights;
	RealMatrix bias;
	Function func;
	int input_shape;
	int numUnits;
	RealMatrix input;
	RealMatrix netSum;
	RealMatrix deltaW;
	RealMatrix deltab;
	
	public void configure(int input_shape) {
		
	}
	
	public RealMatrix forward(RealMatrix X) {
		return X.copy();
	}
	
	public RealMatrix transfer(RealMatrix a) {
		return a.copy();
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
	
	public void setBias(RealMatrix bias) {
		this.bias = bias;
	}
	
	public void setWeigths(RealMatrix weights) {
		this.weights = weights;
	}
	
	
	
	
	
}
