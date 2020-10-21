package losses;

import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.JsonObject;

public class MeanSquaredError extends Loss {
	
	String name = "meanSquaredError";
	
	// Metric method
	public double value(RealMatrix yTrue, RealMatrix yPred)  {
				
		return this.evaluateSum(yTrue, yPred);
	}
	
	// Loss method
	public RealMatrix evaluate(RealMatrix yTrue, RealMatrix yPred) {
		RealMatrix mseMatrix = (yPred.add(yTrue.scalarMultiply(-1.0)));
		
		return mseMatrix;			
	}
	
	// Loss method
	public RealMatrix evaluateDer(RealMatrix yTrue, RealMatrix yPred) {
		return yPred.add(yTrue.scalarMultiply(-1.0)).scalarMultiply(2);
	}

		
	/*
	public double evaluateSum(RealMatrix yTrue, RealMatrix yPred) {
		RealMatrix lossMatrix = evaluate(yTrue, yPred);
		
		double mse = 0.0;
		for (int i=0; i<lossMatrix.getRowDimension(); i++) {
			for (int j=0; j<lossMatrix.getColumnDimension(); j++) {
				mse = mse + Math.pow(lossMatrix.getEntry(i, j),2);
			}	
		}
		
		return mse;
				
	}
	*/
	
	
}
