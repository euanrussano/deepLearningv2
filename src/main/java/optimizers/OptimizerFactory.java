package optimizers;

import com.eclipsesource.json.JsonObject;

public class OptimizerFactory {
	
	public static Optimizer fromJson(JsonObject obj) {
		
		String type = obj.get("type").toString().replace("\"", "");
		
		double learningRate = obj.get("learningRate").asDouble();
		
		switch(type) {
		case ("sgd"):
			return new StochasticGradientDescent(learningRate);
		default:
			return null;
		}
			
	}

}
