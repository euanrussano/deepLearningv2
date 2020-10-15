package datasets;

import java.util.Iterator;

import org.apache.commons.math4.linear.RealMatrix;

public class TableDataset implements Dataset, Iterator<Item>{

	private int currentRow;
	private RealMatrix features;
	private RealMatrix target;
	
	public TableDataset() {
		currentRow = 0;
	}
	
	public TableDataset(RealMatrix features,	RealMatrix target) {
		this.features = features;
		this.target = target;
		currentRow = 0;
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
		Item item = new Item(features.getRowMatrix(currentRow), target.getRowMatrix(currentRow));
		currentRow++;
		return item;
	}

	@Override
	public int getCurrentRow() {
		return currentRow;
	}
}
