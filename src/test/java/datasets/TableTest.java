package datasets;

import org.junit.Test;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;


public class TableTest {

	@Test
    public void createDummyDataset()
    {
		double[][] matrixData = { {1,2,3}, {2,5,3}};
		RealMatrix X = MatrixUtils.createRealMatrix(matrixData);
		
		double[][] matrixData2 = {{1}, {2}};
		RealMatrix Y = MatrixUtils.createRealMatrix(matrixData2);
		
		
		TableDataset dataset = new TableDataset(X, Y);
		
		Item item = dataset.next();
		//System.out.println(item.toString());
		assertTrue(item.getFeatures().equals(X.getRowMatrix(0)));
		assertTrue(item.getTarget().equals(Y.getRowMatrix(0)));
    }
	
	@Test
    public void loadingCSV()
    {
		double[][] matrixData = {{15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22}};
		RealMatrix X = MatrixUtils.createRealMatrix(matrixData);
		
		double[][] matrixData2 = {{1}};
		RealMatrix Y = MatrixUtils.createRealMatrix(matrixData2);
		
		Dataset dataset = TableDataLoader.loadData("C:\\Users\\eruss\\eclipse-workspace\\deepLearningv2\\datasets\\wheat-seeds.csv");
        
        Item item = dataset.next();
		System.out.println(item.toString());
		assertTrue(item.getFeatures().equals(X.getRowMatrix(0)));
		assertTrue(item.getTarget().equals(Y.getRowMatrix(0)));
        
    }
	
	@Test
    public void loadingCSVandIterate()
    {
		double[][] matrixData = {{15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22}};
		RealMatrix X = MatrixUtils.createRealMatrix(matrixData);
		
		double[][] matrixData2 = {{1}};
		RealMatrix Y = MatrixUtils.createRealMatrix(matrixData2);
		
		Dataset dataset = TableDataLoader.loadData("C:\\Users\\eruss\\eclipse-workspace\\deepLearningv2\\datasets\\wheat-seeds.csv");
        
		Item item;
        while (dataset.hasNext()) {
        	item = dataset.next();
        	System.out.println(item.toString());
        }
        
    }
	
}
