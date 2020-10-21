package layers;

import java.util.Arrays;
import java.util.function.Function;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;

import functions.ActivationFunction;
import utils.MatrixOperations;

public class Dense extends Layer{
	
	String type = "dense";
	
	public Dense(int numUnits, ActivationFunction func) {
		super();
		this.numUnits = numUnits;
		this.func = func;
	}
	
	public Dense(int numUnits, ActivationFunction func, int input_shape) {
		this(numUnits, func);
		this.inputShape = input_shape;
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
			bias_mat[0][col] = Math.random();
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
		a = func.evaluate(a);
		
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
	
	@Override
	public RealMatrix backward(RealMatrix accum_grad) {
		
		//System.out.println("a = " + input); // 1 x 2
		
		RealMatrix grad_a2 = MatrixOperations.elementMultiply(func.evaluateDer(netSum),accum_grad);
		//System.out.println("grad_a2 = " + grad_a2);
		//System.out.println("input = " + input);
		
		deltaW = input.transpose().multiply(grad_a2);
		//System.out.println(deltaW);
		deltab = MatrixOperations.ones_like(input).getColumnMatrix(0).transpose().multiply(grad_a2);
		
		
		accum_grad = grad_a2.multiply(weights.transpose());
		//System.out.println(grad_a2.getRowDimension() + " " + grad_a2.getColumnDimension());
		//System.out.println(weights.getRowDimension() + " " + weights.getColumnDimension());
		
		return accum_grad;
	}
}

