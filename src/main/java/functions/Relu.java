package functions;

public class Relu implements Function{
	
	public double evaluate(double z) {
		return z>0 ? z: 0;
	}
	
	public double evaluate_der(double z) {
		return z>0 ? 1: 0;
	}


}
