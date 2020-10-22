package networks;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;

import layers.Layer;
import layers.LayerFactory;
import losses.Loss;
import losses.LossFactory;
import optimizers.Optimizer;
import optimizers.OptimizerFactory;

public class NetworkFactory {
	
	public static Network createFromJSON(String path) {
		// https://stackoverflow.com/questions/2591098/how-to-parse-json-in-java
		/*
		 * {
			  
			  "layers": [ {
			      "type": "Dense",
			      "numUnits": 1,
			      "activation": "relu",
			      "inputShape": 2,
			      "weights": {{0.87}, {-2.3}}
			      "bias": {{0.0}}
			    },
			    {
			      "type": "Dense",
			      "numUnits": 2,
			      "activation": "sigmoid",
			      "inputShape": 10,
			      "weights": {{1.74, -0.76}};
			      "bias": {{0.0, 0.0}};
			    } ],
			  "loss": {
			  	"type":"accuracy",
			  }
			  "optimizer": {
			  	"type": "sgd",
			  	"learningRate": 0.01
			  	}
			}
		 */
		String content;
		JsonObject object = null;
		try {
			content = new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8);
			object = Json.parse(content).asObject();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Optimizer optimizer = OptimizerFactory.fromJson((JsonObject)object.get("optimizer"));
		
		Loss loss = LossFactory.fromJson((JsonObject)object.get("loss"));
		
		JsonArray layersJson= object.get("layers").asArray();
		List<Layer> layers = new ArrayList<>();
		for (JsonValue obj : layersJson) {
			Layer layer = LayerFactory.fromJson((JsonObject)obj);
			
			layers.add(layer);
			
		}
		
		Network model = new Network();
		for(Layer layer : layers) {
			model.add(layer);
		}
		
		model.setOptimizer(optimizer);
		model.setLoss(loss);
		
		return model;
	}
	
}
