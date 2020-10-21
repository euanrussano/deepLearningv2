package functions;

public class Sigmoid extends ActivationFunction{
	
	String name = "sigmoid";
	
	@Override
	public double evaluateSingle(double z) {
		return 1/(1 + Math.exp(-z));
	}
	
	@Override
	public double evaluateDerSingle(double z) {
		return evaluateSingle(z)*(1-evaluateSingle(z));
	}
	
}
