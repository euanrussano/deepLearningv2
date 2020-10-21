package functions;

public class Relu extends ActivationFunction{
	
	String name = "relu";
	
	@Override
	public double evaluateSingle(double z) {
		return z>0 ? z: 0;
	}
	
	@Override
	public double evaluateDerSingle (double z) {
		return z>0 ? 1: 0;
	}

}
