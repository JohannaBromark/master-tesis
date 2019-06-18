package experiments;

import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.DataHandle;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

public abstract class Experiment {


    public static Data initData(String dataset, String[] attributes) throws IOException {
        // Define data
        String datasetPath = ""+dataset; //TODO: add dataset path
        Data data = Data.create( datasetPath+ "/"+dataset+".csv", StandardCharsets.UTF_8, ',');

        for (String attr : attributes) {
            String filePath = datasetPath+"/hierarchy_"+attr+".csv";
            AttributeType.Hierarchy attribute = AttributeType.Hierarchy.create(filePath, StandardCharsets.UTF_8, ';');
            data.getDefinition().setAttributeType(attr, AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
            data.getDefinition().setHierarchy(attr, attribute);
        }

        return data;
    }

    public static Data initData(String dataset, String specific_dataset, String[] attributes) throws IOException {
        // Define data
        String datasetPath = ""+dataset; //TODO: add dataset path
        Data data = Data.create( datasetPath+ "/"+specific_dataset, StandardCharsets.UTF_8, ',');

        for (String attr : attributes) {
            String filePath = datasetPath+"/hierarchy_"+attr+".csv";
            AttributeType.Hierarchy attribute = AttributeType.Hierarchy.create(filePath, StandardCharsets.UTF_8, ';');
            data.getDefinition().setAttributeType(attr, AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
            data.getDefinition().setHierarchy(attr, attribute);
        }

        return data;
    }

    public static void saveToFile(String fileName, DataHandle dataHandle) {
        Iterator<String[]> itHandle = dataHandle.iterator();
        try {
            PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            while (itHandle.hasNext()) {
                String[] row = itHandle.next();
                String line = String.join(",", row);
                writer.write(line+"\n");
            }
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

}
