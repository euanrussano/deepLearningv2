package functions;

public class Linear extends ActivationFunction{
	
	String name = "linear";
	
	@Override
	public double evaluateSingle (double z) {
		return z;
	}
	
	@Override
	public double evaluateDerSingle(double z) {
		return 1;
	}

}
