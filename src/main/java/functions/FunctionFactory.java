package functions;

import com.eclipsesource.json.JsonObject;

public class FunctionFactory {

	public static ActivationFunction fromJson(JsonObject obj) {
		String type = obj.get("type").asString();
		
		ActivationFunction func = new Linear();
		switch (type) {
			case ("Sigmoid"):
				func = new Sigmoid();
			case ("Relu"):
				func = new Relu();
			case ("Linear"):
				func = new Linear();
			}
		
		return func;
	}
}
