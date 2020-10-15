package datasets;

import org.apache.commons.math4.linear.MatrixUtils;
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
		int numCols = features.getColumnDimension();
		
		RealMatrix expanded = MatrixUtils.createRealMatrix(new double[1][numCols+1]);
		
		expanded.setSubMatrix(features.getData(), 0, 0);
		
		expanded.setSubMatrix(target.getData(), 0, numCols);
		
		RealMatrixFormat formatter = new RealMatrixFormat("[", "]\n", "[", "]", "", " ");
		
		return formatter.format(expanded);
	}
	
	

}
