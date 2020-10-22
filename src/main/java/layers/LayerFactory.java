package layers;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;

import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;

import functions.ActivationFunction;
import functions.FunctionFactory;

public class LayerFactory {
	
	public static Layer fromJson(JsonObject obj) {
		String type = obj.get("type").asString().replace("\"", "");
		int numUnits = obj.get("numUnits").asInt();
		
		ActivationFunction func = FunctionFactory.fromJson(obj);
		int inputShape = obj.get("inputShape").asInt();
		JsonArray weightsJson = obj.get("weights").asArray();
		JsonArray biasJson = obj.get("bias").asArray();
		
		int numRows = weightsJson.size();
		int numCols = ((JsonArray)weightsJson.get(0)).size();
		RealMatrix weights = MatrixUtils.createRealMatrix(new double[numRows][numCols]);
		
		JsonArray row;
		for (int i=0; i<numRows; i++) {
			row = (JsonArray)weightsJson.get(i);
			for (int j=0; j<numCols; j++) {
				weights.setEntry(i, j, row.get(j).asDouble());
			}	
		}
		
		numRows = biasJson.size();
		numCols = ((JsonArray)biasJson.get(0)).size();
		RealMatrix bias = MatrixUtils.createRealMatrix(new double[numRows][numCols]);
		
		for (int i=0; i<numRows; i++) {
			row = (JsonArray)biasJson.get(i);
			for (int j=0; j<numCols; j++) {
				bias.setEntry(i, j, row.get(j).asDouble());
			}	
		}
		
		
		switch (type) {
			case("dense"):
				
				Layer dense = new Dense(numUnits, func, inputShape);
				dense.setConfigured(true);
				return dense;
			default:
				return null;
		} 
		
	}
	

}

