package datasets;

import java.util.Iterator;

public interface Dataset extends Iterator<Item> {

	public void batch(int size);
	
	public int getLength();
	
	public void reset();
	
	public int getCurrentRow();
	
	public int getBatchSize();
	
	public int getTargetDimension();
}
