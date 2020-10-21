package NN.deepLearningv2;

import static org.junit.Assert.assertTrue;

import java.io.IOException;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;
import org.apache.commons.math4.linear.RealMatrixFormat;
import org.junit.Test;

import datasets.Dataset;
import datasets.TableDataset;
import functions.Relu;
import functions.Sigmoid;
import layers.Dense;
import layers.Layer;
import losses.Loss;
import losses.MeanSquaredError;
import losses.Metric;
import networks.Network;
import optimizers.Optimizer;
import optimizers.StochasticGradientDescent;

/**
 * Unit test for simple App.
 */
public class AppTest 
{
    private Dataset generateData() {
    	
    	RealMatrix X = MatrixUtils.createRealMatrix(new double[][] {{1.62, -0.61}, {-0.53, -1.07}});
		RealMatrix y = MatrixUtils.createRealMatrix(new double[][] {{0.87, -2.3}, {1.74, -0.76}});
		
		Dataset dataset = new TableDataset(X, y);
		dataset.batch(2);
		
		return dataset;
    }
	/**
     * Rigorous Test :-)
     */
    @Test
    public void testPredict()
    {
    	Dataset dataset = generateData();
		
		Optimizer sgd = new StochasticGradientDescent(0.5);
		Loss mse = new MeanSquaredError();
		Loss mse_metric = new MeanSquaredError();
		
		Network model = new Network();
		Layer l1 = new Dense(1,new Relu(),2);
		Layer l2 = new Dense(2,new Relu());
			
		// hard set the weights
		
		RealMatrix weights1 = MatrixUtils.createRealMatrix(new double[][] {{0.32}, {-0.25}});
		RealMatrix bias1 = MatrixUtils.createRealMatrix(new double[][] {{0.0}});
						
		RealMatrix weights2 = MatrixUtils.createRealMatrix(new double[][] {{1.46, -2.06}});
		RealMatrix bias2 = MatrixUtils.createRealMatrix(new double[][] {{0.0, 0.0}});
		
		
		model.add(l1);
		model.add(l2);
		model.compile(sgd, mse, mse_metric);
		
		l1.setWeigths(weights1);
		l1.setBias(bias1);
		l2.setWeigths(weights2);
		l2.setBias(bias2);
		
		RealMatrix yPred = model.predict(dataset);
		
		System.out.println("yPred = "+ yPred);
    }
    
    @Test
    public void testTraining()
    {	
		Dataset dataset = generateData();
		
		Optimizer sgd = new StochasticGradientDescent(1e-1);
		Loss mse = new MeanSquaredError();
		Loss mse_metric = new MeanSquaredError();
		
		Network model = new Network();
		Layer l1 = new Dense(1,new Relu(),2);
		Layer l2 = new Dense(2,new Sigmoid());
			
		// hard set the weights
		
		RealMatrix weights1 = MatrixUtils.createRealMatrix(new double[][] {{0.87}, {-2.3}});
		RealMatrix bias1 = MatrixUtils.createRealMatrix(new double[][] {{0.0}});
		
				
		RealMatrix weights2 = MatrixUtils.createRealMatrix(new double[][] {{1.74, -0.76}});
		RealMatrix bias2 = MatrixUtils.createRealMatrix(new double[][] {{0.0, 0.0}});
		
		
		model.add(l1);
		model.add(l2);
		model.compile(sgd, mse, mse_metric);
		
		l1.setWeigths(weights1);
		l1.setBias(bias1);
		l2.setWeigths(weights2);
		l2.setBias(bias2);
		
		model.fit(dataset, 10);
		
		RealMatrix yPred = model.predict(dataset);
		
		System.out.println("yPred = "+ yPred);
    }
    
    @Test
    public void testTrainingStochastic()
    {	
		Dataset dataset = generateData();
		dataset.batch(1);
		
		Optimizer sgd = new StochasticGradientDescent(1e-1);
		Loss mse = new MeanSquaredError();
		Loss mse_metric = new MeanSquaredError();
		
		Network model = new Network();
		Layer l1 = new Dense(1,new Relu(),2);
		Layer l2 = new Dense(2,new Sigmoid());
			
		// hard set the weights
		
		RealMatrix weights1 = MatrixUtils.createRealMatrix(new double[][] {{0.87}, {-2.3}});
		RealMatrix bias1 = MatrixUtils.createRealMatrix(new double[][] {{0.0}});
		
				
		RealMatrix weights2 = MatrixUtils.createRealMatrix(new double[][] {{1.74, -0.76}});
		RealMatrix bias2 = MatrixUtils.createRealMatrix(new double[][] {{0.0, 0.0}});
		
		
		model.add(l1);
		model.add(l2);
		model.compile(sgd, mse, mse_metric);
		
		l1.setWeigths(weights1);
		l1.setBias(bias1);
		l2.setWeigths(weights2);
		l2.setBias(bias2);
		
		model.fit(dataset, 100);
		
		RealMatrix yPred = model.predict(dataset);
		
		System.out.println("yPred = "+ yPred);
    }
    
    @Test
    public void testTrainingDeeper()
    {
		Dataset dataset = generateData();
		
		Optimizer sgd = new StochasticGradientDescent(1e-1);
		Loss mse = new MeanSquaredError();
		Loss mse_metric = new MeanSquaredError();
		
		Network model = new Network();
		Layer l1 = new Dense(1,new Relu(),2);
		Layer l2 = new Dense(2,new Relu(),2);
		Layer l3 = new Dense(2,new Sigmoid());
		
		model.add(l1);
		model.add(l2);
		model.add(l3);
		model.compile(sgd, mse, mse_metric);
		
		
		model.fit(dataset, 100);
		
		RealMatrix yPred = model.predict(dataset);
		
		System.out.println("yPred = "+ yPred);
    }
    
    @Test
    public void testExportNetwork()
    {
		Dataset dataset = generateData();
		
		Optimizer sgd = new StochasticGradientDescent(1e-1);
		Loss mse = new MeanSquaredError();
		Loss mse_metric = new MeanSquaredError();
		
		Network model = new Network();
		Layer l1 = new Dense(1,new Relu(),2);
		Layer l2 = new Dense(2,new Relu(),2);
		Layer l3 = new Dense(2,new Sigmoid());
		
		model.add(l1);
		model.add(l2);
		model.add(l3);
		model.compile(sgd, mse, mse_metric);
		
		
		model.fit(dataset, 10);
		
		try {
			model.saveAsJson();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
    }
    
    @Test
    public void testImportNetwork()
    {
		Network model = new Network();
		
		model.createFromJSON("C:\\Users\\eruss\\eclipse-workspace\\deepLearningv2\\model.json");
		
    }
}
