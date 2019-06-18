package experiments;

import org.deidentifier.arx.*;
import org.deidentifier.arx.criteria.EDDifferentialPrivacy;
import org.deidentifier.arx.metric.Metric;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * Test 2.1: different number of attributes in the dataset.
 *
 * */

public class Experiment3 extends Experiment {

    public static void main(String[] args) throws IOException {
        double delta = 1e-4;
        run("musk", delta, new int[]{10, 20});
    }

    public static void run(String dataset, double delta, int[] attributeRange) throws IOException{

        String outputPath = ""; //TODO: add output paht
        int iterations = 25;

        double epsilon = 1d;

        int steps = 0;

        Metric metric = Metric.createLossMetric();
        System.out.println("--------------------------------- Running test 2.1");
        for (int i = attributeRange[0]; i <= attributeRange[1]; i++) {
            File folder = new File(""+i); // TODO: add subfile path
            File[] fileList = folder.listFiles();
            int fileidx = 0;
            for (File filename : fileList) {
                String[] filenameParts = filename.toString().split("/");
                String file = filenameParts[filenameParts.length - 1];

                String[] attributes = file.split("-");
                attributes[attributes.length-1] = attributes[attributes.length-1].replace(".csv", "");

                if (steps < 300){
                    int maxCombinations = computeMaxCombinations(dataset, attributes);
                    steps = maxCombinations > 300? 300 : maxCombinations-1;
                }

                for (int iter=0; iter<iterations; iter++) {
                    String filePath = outputPath+"/"+i+"/"+String.join("-", attributes)+"_"+(iter+1)+".csv";
                    Data data = initData(dataset, "attribute_subsets/"+attributes.length+"/"+file, attributes);
                    ARXAnonymizer anonymizer = new ARXAnonymizer();
                    EDDifferentialPrivacy criterion = new EDDifferentialPrivacy(epsilon, delta);
                    ARXConfiguration config = ARXConfiguration.create();

                    config.setQualityModel(metric);
                    config.addPrivacyModel(criterion);
                    config.setSuppressionLimit(1d);  //Not sure if this need to be specified or how it affects the result
                    config.setHeuristicSearchStepLimit(steps);
                    config.setHeuristicSearchStepSemantics(ARXConfiguration.SearchStepSemantics.EXPANSIONS);
                    ARXResult result = anonymizer.anonymize(data, config);
                    DataHandle optimal = result.getOutput();
                    saveToFile(filePath, optimal);
                }
                fileidx ++;
                System.out.println(fileidx + " of "+fileList.length+" done");
            }
            System.out.println(i + " attributes done");
        }

    }

    private static int computeMaxCombinations(String dataset, String[] attributes) throws IOException {
        ArrayList<Integer> hierarchyLevels = new ArrayList<>();
        String path = ""+dataset; //TODO: add dataset path
        for (String attribute : attributes){
            BufferedReader reader = new BufferedReader(new FileReader(path+"/hierarchy_"+attribute+".csv"));
            String line = reader.readLine();
            String[] levels = line.split(";");
            hierarchyLevels.add(levels.length);
        }

        List<List<String>> allLevels = new ArrayList<>();

        for (int l : hierarchyLevels) {
            List<String> levels = new ArrayList<>();
            for (int i = 0; i < l; i++){
                levels.add(i+"");
            }
            allLevels.add(levels);
        }

        List<String> combinations = new ArrayList<>();

        GeneratePermutations(allLevels, combinations, 0, "");

        return combinations.size();
    }

    private static void GeneratePermutations(List<List<String>> Lists, List<String> result, int depth, String current)
    {
        if(depth == Lists.size())
        {
            result.add(current);
            return;
        }

        for(int i = 0; i < Lists.get(depth).size(); ++i)
        {
            GeneratePermutations(Lists, result, depth + 1, current + Lists.get(depth).get(i));
        }
    }

}
