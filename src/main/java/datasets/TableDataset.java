package datasets;

import java.util.Iterator;

import org.apache.commons.math4.linear.RealMatrix;

public class TableDataset implements Dataset, Iterator<Item>{

	private int currentRow;
	private RealMatrix features;
	private RealMatrix target;
	private int batchSize = 1;
	
	public TableDataset() {
		currentRow = 0;
	}
	
	public TableDataset(RealMatrix features,	RealMatrix target) {
		this.features = features;
		this.target = target;
		currentRow = 0;
	}
	
	public void batch(int size) {
		this.batchSize = size;
	}
			
	@Override	
	public void reset() {
		currentRow = 0;
	}

	@Override
	public int getLength() {
		
		return features.getRowDimension();
	}
	
	public int getNumFeatures() {
		return features.getColumnDimension();
	}

	@Override
	public boolean hasNext() {
		if (currentRow < features.getRowDimension()) {
			return true;
		}
		return false;
	}

	@Override
	public Item next() {
		Item item;
		
		int N = features.getRowDimension()-currentRow;
		if (batchSize == 1) {
			item = new Item(features.getRowMatrix(currentRow), target.getRowMatrix(currentRow));
		}else {
			if (currentRow + batchSize-1 < features.getRowDimension()) {
				
				item = new Item(features.getSubMatrix(currentRow, currentRow + batchSize-1, 0, features.getColumnDimension()-1),
								   target.getSubMatrix(currentRow, currentRow + batchSize-1, 0, target.getColumnDimension()-1));
			} else {
				
				item = new Item(features.getSubMatrix(currentRow, currentRow + N-1, 0, features.getColumnDimension()-1),
						   target.getSubMatrix(currentRow, currentRow + N-1, 0, target.getColumnDimension()-1));
			}
		}
		if (batchSize <= N) { 
			currentRow = currentRow + batchSize;
		} else {
			currentRow = currentRow + N;
		}
		
		//System.out.println("Item = " + item);
		//System.out.println("currentRow = " + currentRow);
		return item;
	}

	@Override
	public int getCurrentRow() {
		return currentRow;
	}

	@Override
	public int getBatchSize() {
		return this.batchSize;
	}
	
	public int getTargetDimension() {
		return target.getColumnDimension();
	}
}
