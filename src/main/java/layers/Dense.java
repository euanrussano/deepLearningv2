package layers;

import java.util.Arrays;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;
import org.apache.commons.math4.linear.RealMatrixFormat;

import functions.Function;

public class Dense extends Layer{
	
	public Dense(int numUnits, Function func) {
		super();
		this.numUnits = numUnits;
		this.func = func;
	}
	
	public Dense(int numUnits, Function func, int input_shape) {
		this(numUnits, func);
		this.input_shape = input_shape;
	}
	
	public void configure(int input_shape) {
		int numRows = input_shape;
		int numCols = numUnits;
		
		double[][] weights_mat = new double[numRows][numCols];
		double[][] bias_mat = new double[1][numCols];
		
		for (int col = 0; col < numCols; col++) {
			for (int row = 0; row < numRows; row++) {
		        weights_mat[row][col] = Math.random();
		    }
			weights_mat[0][col] = Math.random();
		}
		
		weights = MatrixUtils.createRealMatrix(weights_mat);
		bias = MatrixUtils.createRealMatrix(bias_mat);
		 
	}
	
	
	@Override
	public RealMatrix forward(RealMatrix X) {
		
		// adjust the bias vector to a matrix
		RealMatrix biasMatrix = transformBiasToMatrix(X.getRowDimension());
		
		input = X;
		RealMatrix a = X.multiply(weights).add(biasMatrix); // 1 x 3 x 3 x 10 + 1 x 10 =1 x 10
		netSum = a;
		a = transfer(a);
		
		// store internally the current output
		return a;
	}
	
	public RealMatrix transformBiasToMatrix(int numRows) {
		double[][] onesVec = new double[1][numRows];
		for (double[] row: onesVec)
		    Arrays.fill(row, 1.0);
		
		RealMatrix onesMat= MatrixUtils.createRealMatrix(onesVec);
		
		return bias.transpose().multiply(onesMat).transpose();
		
	}
	
	public RealMatrix transfer(RealMatrix a) {
		int numRows = a.getRowDimension();
		int numColumns = a.getColumnDimension();
		
		RealMatrix z = a.copy();
		for (int i=0; i<numRows; i++) {
			for (int j=0; j<numColumns; j++) {
				double value_a = a.getEntry(i, j);
				z.setEntry(i, j, func.evaluate(value_a));
			}	
		} 
		
		return z;
	}
	
	public RealMatrix transfer_der(RealMatrix a) {
		int numRows = a.getRowDimension();
		int numColumns = a.getColumnDimension();
		
		RealMatrix z = a.copy();
		for (int i=0; i<numRows; i++) {
			for (int j=0; j<numColumns; j++) {
				double value_a = a.getEntry(i, j);
				z.setEntry(i, j, func.evaluate_der(value_a));
			}	
		} 
		
		return z;
	}
	
	public RealMatrix elementMultiply(RealMatrix A, RealMatrix B) {
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
	@Override
	public RealMatrix backward(RealMatrix accum_grad) {
		
		//System.out.println("a = " + input); // 1 x 2
		RealMatrix grad_a2 = elementMultiply(transfer_der(netSum),accum_grad);
		//System.out.println("grad_a2 = " + grad_a2);
		deltaW = input.transpose().multiply(grad_a2);
		deltab = grad_a2;
		
		accum_grad = grad_a2.multiply(weights.transpose());
		
		
		return accum_grad;
	}

}

