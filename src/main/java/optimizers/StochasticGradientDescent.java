package optimizers;
import java.util.List;

import org.apache.commons.math4.linear.RealMatrix;


public class StochasticGradientDescent extends Optimizer{
	
	public StochasticGradientDescent(double lr) {
		super(lr);
		setType("sgd");
	}

	public RealMatrix updateSingle(RealMatrix X, RealMatrix dX) {
		return X.add(dX.scalarMultiply(-1*this.learningRate));
	}
	

	
}
