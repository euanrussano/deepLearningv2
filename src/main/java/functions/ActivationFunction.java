package functions;

import org.apache.commons.math4.linear.RealMatrix;

public abstract class ActivationFunction {
	
	private String type;
	
	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public double evaluateSingle(double z) {return z;}
	
	public double evaluateDerSingle(double z) {return z;}
	
	public RealMatrix evaluateDer(RealMatrix a) {
		int numRows = a.getRowDimension();
		int numColumns = a.getColumnDimension();
		
		RealMatrix z = a.copy();
		for (int i=0; i<numRows; i++) {
			for (int j=0; j<numColumns; j++) {
				double value_a = a.getEntry(i, j);
				z.setEntry(i, j, evaluateDerSingle(value_a));
			}	
		} 
		
		return z;
	}
	
	public RealMatrix evaluate(RealMatrix a) {
		int numRows = a.getRowDimension();
		int numColumns = a.getColumnDimension();
		
		RealMatrix z = a.copy();
		for (int i=0; i<numRows; i++) {
			for (int j=0; j<numColumns; j++) {
				double value_a = a.getEntry(i, j);
				z.setEntry(i, j, evaluateSingle(value_a));
			}	
		} 
		
		return z;
	}

	public String toJson() {
		return type;
	}
}
