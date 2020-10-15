package layers;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;
import org.apache.commons.math4.linear.RealMatrixFormat;
import org.junit.Test;

import functions.Linear;
import functions.Relu;
import functions.Sigmoid;

public class DenseTest {
	
	@Test
	public void testForward() {
		
		double[][] matrixData = {{1,0}};
		RealMatrix X = MatrixUtils.createRealMatrix(matrixData);
		
		Layer l1 = new Dense(1,new Sigmoid(),2);
		Layer l2 = new Dense(2,new Sigmoid());
		
		l2.configure(l1.getNumUnits());
		
		// hard set the weights
		
		RealMatrix weights1 = MatrixUtils.createRealMatrix(new double[][] {{0.134}, {0.847}});
		RealMatrix bias1 = MatrixUtils.createRealMatrix(new double[][] {{0.764}});
		l1.setWeigths(weights1);
		l1.setBias(bias1);
				
		RealMatrix weights2 = MatrixUtils.createRealMatrix(new double[][] {{0.255, 0.449}});
		RealMatrix bias2 = MatrixUtils.createRealMatrix(new double[][] {{0.495, 0.651}});
		
		RealMatrixFormat formatter = new RealMatrixFormat("[", "]\n", "[", "]", "", " ");
		
		l2.setWeigths(weights2);
		l2.setBias(bias2);
		
		
		
		RealMatrix h1 = l1.forward(X); // 1x2 * 2x1 = 1x1
		//System.out.println("h1 = "+ formatter.format(h1)); // h1 = [[0,7105383287]]
		
		RealMatrix out = l2.forward(h1); // 1x1 * 1x2
		//System.out.println("out = "+ formatter.format(out)); // out = [[0,6628872059 0,7251258181]]
	}
	
	// backward test generate with example warm up numpy from pytorch using numpy seed 1
	// the script used to double check the values is "Backpropagation Dense.ipynb"
	@Test
	public void testBackwardOneSample() {
		 
		RealMatrix X = MatrixUtils.createRealMatrix(new double[][] {{1.62, -0.61}});
		RealMatrix y = MatrixUtils.createRealMatrix(new double[][] {{-0.53, -1.07}});
		
		Layer l1 = new Dense(1,new Relu(),2);
		Layer l2 = new Dense(2,new Relu());
		
		l2.configure(l1.getNumUnits());
		
		// hard set the weights
		
		RealMatrix weights1 = MatrixUtils.createRealMatrix(new double[][] {{0.87}, {-2.3}});
		RealMatrix bias1 = MatrixUtils.createRealMatrix(new double[][] {{0.0}});
		l1.setWeigths(weights1);
		l1.setBias(bias1);
				
		RealMatrix weights2 = MatrixUtils.createRealMatrix(new double[][] {{1.74, -0.76}});
		RealMatrix bias2 = MatrixUtils.createRealMatrix(new double[][] {{0.0, 0.0}});
		l2.setWeigths(weights2);
		l2.setBias(bias2);
		
		RealMatrixFormat formatter = new RealMatrixFormat("[", "]\n", "[", "]", "", " ");
		
		// forward pass
		RealMatrix h_relu = l1.forward(X); // 1x2 * 2x1 = 1x1
		RealMatrix y_pred = l2.forward(h_relu); // 1x1 * 1x2
		
		System.out.println("y_pred = " + y_pred + "\n");
		// backward pass
		RealMatrix loss_grad = y_pred.add(y.scalarMultiply(-1.0)).scalarMultiply(2);		
		System.out.println("loss_grad = " + loss_grad + "\n");
		
		RealMatrix deltaout = l2.backward(loss_grad);
		System.out.println("dw2 = "+ formatter.format(l2.deltaW));
		System.out.println("grad_h_relu = "+ formatter.format(deltaout));
		deltaout = l1.backward(deltaout);
		System.out.println("dw1 = "+ formatter.format(l1.deltaW));
		System.out.println("grad_w1 = "+ formatter.format(deltaout));
	}
	
	// backward test generate with example warm up numpy from pytorch using numpy seed 1
		// the script used to double check the values is "Backpropagation Dense.ipynb"
		@Test
		public void testBackwardTwoSample() {
			 
			RealMatrix X = MatrixUtils.createRealMatrix(new double[][] {{1.62, -0.61}, {-0.53, -1.07}});
			RealMatrix y = MatrixUtils.createRealMatrix(new double[][] {{0.87, -2.3}, {1.74, -0.76}});
			
			Layer l1 = new Dense(1,new Relu(),2);
			Layer l2 = new Dense(2,new Relu());
			
			l2.configure(l1.getNumUnits());
			
			// hard set the weights
			
			RealMatrix weights1 = MatrixUtils.createRealMatrix(new double[][] {{0.32}, {-0.25}});
			RealMatrix bias1 = MatrixUtils.createRealMatrix(new double[][] {{0.0}});
			l1.setWeigths(weights1);
			l1.setBias(bias1);
					
			RealMatrix weights2 = MatrixUtils.createRealMatrix(new double[][] {{1.46, -2.06}});
			RealMatrix bias2 = MatrixUtils.createRealMatrix(new double[][] {{0.0, 0.0}});
			l2.setWeigths(weights2);
			l2.setBias(bias2);
			
			RealMatrixFormat formatter = new RealMatrixFormat("[", "]\n", "[", "]", "", " ");
			
			// forward pass
			RealMatrix h_relu = l1.forward(X); // 1x2 * 2x1 = 1x1
			RealMatrix y_pred = l2.forward(h_relu); // 1x1 * 1x2
			
			System.out.println("y_pred = " + y_pred + "\n");
			// backward pass
			RealMatrix loss_grad = y_pred.add(y.scalarMultiply(-1.0)).scalarMultiply(2);		
			System.out.println("loss_grad = " + loss_grad + "\n");
			
			RealMatrix deltaout = l2.backward(loss_grad);
			System.out.println("dw2 = "+ formatter.format(l2.deltaW));
			System.out.println("grad_h_relu = "+ formatter.format(deltaout));
			deltaout = l1.backward(deltaout);
			System.out.println("dw1 = "+ formatter.format(l1.deltaW));
			System.out.println("grad_w1 = "+ formatter.format(deltaout));
		}
		
		// backward test generate with example warm up numpy from pytorch using numpy seed 1
		// the script used to double check the values is "Backpropagation Dense.ipynb"
		@Test
		public void testBackwardTwoSamplewithBias() {
			 
			RealMatrix X = MatrixUtils.createRealMatrix(new double[][] {{1.62, -0.61}});
			RealMatrix y = MatrixUtils.createRealMatrix(new double[][] {{-0.53, -1.07}});
			
			Layer l1 = new Dense(1,new Relu(),2);
			Layer l2 = new Dense(2,new Sigmoid());
			
			l2.configure(l1.getNumUnits());
			
			// hard set the weights
			
			RealMatrix weights1 = MatrixUtils.createRealMatrix(new double[][] {{0.87}, {-2.3}});
			RealMatrix bias1 = MatrixUtils.createRealMatrix(new double[][] {{0.0}});
			l1.setWeigths(weights1);
			l1.setBias(bias1);
					
			RealMatrix weights2 = MatrixUtils.createRealMatrix(new double[][] {{1.74, -0.76}});
			RealMatrix bias2 = MatrixUtils.createRealMatrix(new double[][] {{0.0, 0.0}});
			l2.setWeigths(weights2);
			l2.setBias(bias2);
			
			RealMatrixFormat formatter = new RealMatrixFormat("[", "]\n", "[", "]", "", " ");
			
			// forward pass
			RealMatrix h_relu = l1.forward(X); // 1x2 * 2x1 = 1x1
			RealMatrix y_pred = l2.forward(h_relu); // 1x1 * 1x2
			
			System.out.println("y_pred = " + y_pred + "\n");
			// backward pass
			RealMatrix loss_grad = y_pred.add(y.scalarMultiply(-1.0)).scalarMultiply(2);		
			System.out.println("loss_grad = " + loss_grad + "\n");
			
			RealMatrix deltaout = l2.backward(loss_grad);
			System.out.println("dw2 = "+ formatter.format(l2.deltaW));
			System.out.println("db2 = "+ formatter.format(l2.deltab));
			System.out.println("grad_h_relu = "+ formatter.format(deltaout));
			deltaout = l1.backward(deltaout);
			System.out.println("dw1 = "+ formatter.format(l1.deltaW));
			System.out.println("db1 = "+ formatter.format(l1.deltab));
			System.out.println("grad_w1 = "+ formatter.format(deltaout));
			
		}

}
