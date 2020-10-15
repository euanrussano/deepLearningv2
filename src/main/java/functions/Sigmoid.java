package functions;

public class Sigmoid implements Function{
	
	public double evaluate(double z) {
		return 1/(1 + Math.exp(-z));
	}
	
	public double evaluate_der(double z) {
		return evaluate(z)*(1-evaluate(z));
	}


}
