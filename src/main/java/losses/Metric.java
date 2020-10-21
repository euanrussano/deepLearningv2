package losses;

import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.JsonObject;

public abstract class Metric {
	
	String name;
	
	public abstract double value(RealMatrix yTrue, RealMatrix yPred);
	
	public String toJson() {
		return name;
	}

}
