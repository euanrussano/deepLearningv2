package utils;

import org.apache.commons.math4.linear.RealMatrix;

public class MatrixOperations {

	
	public static RealMatrix elementMultiply(RealMatrix A, RealMatrix B) {
		int numRows = A.getRowDimension();
		int numColumns = A.getColumnDimension();
		
		RealMatrix z = A.copy();
		for (int i=0; i<numRows; i++) {
			for (int j=0; j<numColumns; j++) {
				double value_a = A.getEntry(i, j);
				double value_b = B.getEntry(i, j);
				z.setEntry(i, j, value_a*value_b);
			}	
		} 
		
		return z;
	}
	
	public static RealMatrix ones_like(RealMatrix mat) {
		RealMatrix onesmat = mat.copy().scalarMultiply(0.0).scalarAdd(1.0);
		return onesmat;
	}
	
}
