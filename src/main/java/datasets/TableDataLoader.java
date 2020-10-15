package datasets;

import java.io.FileReader;
import java.util.List;

import org.apache.commons.math4.linear.MatrixUtils;
import org.apache.commons.math4.linear.RealMatrix;

import com.opencsv.CSVReader;

public class TableDataLoader {
	
	public static Dataset loadData(String pathToCsv){
		Dataset dataset = new TableDataset();
		try { 
			  
	        // Create an object of filereader 
	        // class with CSV file as a parameter. 
	        FileReader filereader = new FileReader(pathToCsv); 
	  
	        // create csvReader object passing 
	        // file reader as a parameter 
	        CSVReader csvReader = new CSVReader(filereader); 
	        String[] nextRecord; 
	        
	        List<String[]> allFileRows = csvReader.readAll();
	        
	        int numRows = allFileRows.size();
	        int numCols = allFileRows.get(0).length-1;
	        
	        double[][] Xmatrix = new double[numRows][numCols];
	        double[][] Ymatrix = new double[numRows][1];
	        
	        int row = 0;
	        for(String[] fileRow : allFileRows){
	        	for (int col = 0; col<fileRow.length-1; col++) {
	        		Xmatrix[row][col] = Double.parseDouble(fileRow[col]); 
	        		
	        	}
	        	Ymatrix[row][0] = Double.parseDouble(fileRow[fileRow.length-1]);
	            row ++;
	         }
	    
		
		RealMatrix X = MatrixUtils.createRealMatrix(Xmatrix);
		RealMatrix Y = MatrixUtils.createRealMatrix(Ymatrix);
		
		dataset = new TableDataset(X, Y);
		
		
		
		} catch (Exception e) {
	        e.printStackTrace(); 
	    }
		
		return dataset;
	}

}
