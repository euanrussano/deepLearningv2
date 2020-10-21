package losses;

import org.apache.commons.math4.linear.RealMatrix;

public abstract class Loss {
	
	String name;
	
	public RealMatrix evaluate(RealMatrix yTrue, RealMatrix yPred) {return null;}
	
	public double evaluateSum(RealMatrix yTrue, RealMatrix yPred) {
		RealMatrix lossMatrix = evaluate(yTrue, yPred);
		
		double mse = 0.0;
		for (int i=0; i<lossMatrix.getRowDimension(); i++) {
			for (int j=0; j<lossMatrix.getColumnDimension(); j++) {
				mse = mse + lossMatrix.getEntry(i, j);
			}	
		}
		
		return mse;
				
	}
	
	public RealMatrix evaluateDer(RealMatrix yTrue, RealMatrix yPred) {return null;}
	
	public String toJson() {
		return name;
	}
}
