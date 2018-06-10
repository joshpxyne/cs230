import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Iterator;

import org.json.JSONException;
import org.json.JSONObject;

public class JsonDirectoryConfigurator {
	public static void main(String[] args) {
	JSONObject obj = null;
	String character = "";
	// use whatever parameter you want to use
	String category = "Char";
	// replace with your own directory path
	String newDirPath = "/Users/joshpayne1/desktop/tensorflow-c/";
	// replace with the directory path on your computer
	String datasetPath = "/Users/joshpayne1/downloads/dataset/";	
	// json string generated for dataset
	
	int i = 0;
	
	final int TRAIN_VALIDATE_RATIO = 4;
	
	String str = "";
	try {
		str = readFile(datasetPath+"labels.json",Charset.forName("UTF-8"));
	} catch (IOException e1) {
		e1.printStackTrace();
	}
	try {
		obj = new JSONObject(str);
	} catch (JSONException e) {
		e.printStackTrace();
	}
	File trainDir = new File(newDirPath+"train/");
	if (!trainDir.exists()) {
	    try {
	        trainDir.mkdir();
	    } catch (SecurityException se) {
	    	
	    }
	}
	File validateDir = new File(newDirPath+"validate/");
	if (!validateDir.exists()) {
	    try {
	        validateDir.mkdir();
	    } catch (SecurityException se) {
	    	
	    }
	}

	Iterator<?> keys = obj.keys();
	while (keys.hasNext()) {
		i++;
		String key = (String)keys.next();
		if (i%TRAIN_VALIDATE_RATIO!=0) {
			try {
				Object x = obj.getJSONObject(key);
				String n = x.toString();
				JSONObject subObj = null;
				try {
					subObj = new JSONObject(n);
				} catch (JSONException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				character = subObj.getString(category);
				File theDir = new File(newDirPath+"train/"+character);
				if (!theDir.exists()) {
				    try{
				        theDir.mkdir();
				        // to move (save space), use Files.move
				        
				        	Files.copy(Paths.get(datasetPath+key), Paths.get(newDirPath+"train/"+character+"/"+key), StandardCopyOption.REPLACE_EXISTING);
				    
				    } 
				    catch(SecurityException se){
				      
				    } catch (IOException e) {
						e.printStackTrace();
					}        
				} else {
					try {
						File check = new File(newDirPath+"train/"+character+"/"+key);
			        	if (!check.exists()) Files.copy(Paths.get(datasetPath+key), Paths.get(newDirPath+"train/"+character+"/"+key), StandardCopyOption.REPLACE_EXISTING);
					} catch(SecurityException se) {
						
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				
			} catch (JSONException e) {
				e.printStackTrace();
			}
		}
		else {
			try {
				Object x = obj.getJSONObject(key);
				String n = x.toString();
				JSONObject subObj = null;
				try {
					subObj = new JSONObject(n);
				} catch (JSONException e) {
					e.printStackTrace();
				}
				character = subObj.getString(category);
				File theDir = new File(newDirPath+"validate/"+character);
				if (!theDir.exists()) {
				    try{
				        theDir.mkdir();
				        // to move (save space), use Files.move
				        
				        	Files.copy(Paths.get(datasetPath+key), Paths.get(newDirPath+"validate/"+character+"/"+key), StandardCopyOption.REPLACE_EXISTING);
				    
				    } 
				    catch(SecurityException se){
				    	
				    } catch (IOException e) {
						e.printStackTrace();
					}        
				} else {
					try {
						File check = new File(newDirPath+"validate/"+character+"/"+key);
			        	if (!check.exists()) Files.copy(Paths.get(datasetPath+key), Paths.get(newDirPath+"validate/"+character+"/"+key), StandardCopyOption.REPLACE_EXISTING);
					} catch(SecurityException se) {
						
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				
			} catch (JSONException e) {
				e.printStackTrace();
			}
		}
	}
	System.out.println("Done.");
	}
	static String readFile(String path, Charset encoding) 
			  throws IOException 
			{
			  byte[] encoded = Files.readAllBytes(Paths.get(path));
			  return new String(encoded, encoding);
			}
}
