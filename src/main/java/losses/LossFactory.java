package losses;

import com.eclipsesource.json.JsonObject;

public class LossFactory {
	
	public static Loss fromJson(JsonObject obj) {
		
		String lossType = obj.get("type").toString().replace("\"","");
		
		switch(lossType) {
			case("mse"):
				return new MeanSquaredError();
			default:
				return null;
		}
	}

}
