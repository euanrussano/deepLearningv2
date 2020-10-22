package losses;

import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonObject;

public class Accuracy {
	
	public Accuracy(){
		super();
		//setType("accuracy");
	}
	public double value(RealMatrix yTrue, RealMatrix yPred) {
		
		int correct = 0;
		for (int row=0; row<yPred.getRowDimension(); row++) {
			if (yPred.getEntry(row, 0) == (yTrue.getEntry(row, 0))) {
				correct += 1;
			}	
		}
		
		return correct/yTrue.getRowDimension();	
	}
	
	public JsonObject toJson() {
		JsonObject metric = Json.object().add("name", "accuracy");
		
		return metric;
	}

}
