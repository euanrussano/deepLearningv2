package datasets;

import java.util.Iterator;

public interface Dataset extends Iterator<Item> {

	public int getLength();
	
	public void reset();
	
	public int getCurrentRow();
}
