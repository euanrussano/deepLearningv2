package datasets;

import org.apache.commons.math4.linear.RealMatrix;
import org.apache.commons.math4.linear.RealMatrixFormat;

public class Item {
	
	private RealMatrix features;
	private RealMatrix target;
	
	
	
	public Item(RealMatrix features, RealMatrix target) {
		super();
		this.features = features;
		this.target = target;
	}
	
	public RealMatrix getFeatures() {
		return features;
	}
	public void setFeatures(RealMatrix features) {
		this.features = features;
	}
	public RealMatrix getTarget() {
		return target;
	}
	public void setTarget(RealMatrix target) {
		this.target = target;
	}
	
	public String toString() {
		
		RealMatrixFormat formatter = new RealMatrixFormat("[", "]\n", "[", "]", "", " ");
		
		return formatter.format(this.features) + formatter.format(this.target);
	}
	
	

}
