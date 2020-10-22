package losses;

import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonObject;

public abstract class Loss {
	
	String type;
	
	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

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
	
	public JsonObject toJson() {
		JsonObject obj = Json.object()
				.add("type", type);
		
		return obj;
	}
}
