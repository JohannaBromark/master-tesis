package experiments;

import org.deidentifier.arx.*;
import org.deidentifier.arx.criteria.EDDifferentialPrivacy;
import org.deidentifier.arx.metric.Metric;

import java.io.IOException;
import java.util.HashMap;

public class Experiment4 extends Experiment {

    public static void main(String[] args) throws IOException {
        String[] caf_attributes = new String[]{"age_years", "body_pain","body_temperature","body_weight_kg",
                "cold_and_flu_symptoms", "cold_and_flu_symptoms_duration", "medicine_allergy", "sex", "take_medicines",
                "skin_rash_symptom","works_with_children_elderly_or_sick_people"};

        String[] housing_attributes = new String[]{"longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
                "population","households","median_income","median_house_value"};

        //String[] attributes = new String[20];
        //for (int i = 1; i <= 20; i++) {
        //    attributes[i-1] = Integer.toString(i);
        //}

        double delta = 1e-4;

        //run("caf_small", caf_attributes, delta);
        run("housing_small", housing_attributes, delta);
    }

    public static void run(String dataset, String[] attributes, double delta) throws IOException {

        int iterations = 25;
        int steps = 300;
        String outputPath = ""; //TODO: add outputpath
        HashMap<String, Metric> metrics = new HashMap<>();
        metrics.put("granularity", Metric.createLossMetric());

        double[] epsilons = {2d, 1d};

        System.out.println("--------------------------------- Running test 3.1");

        for (String metricName : metrics.keySet()) {
            for (double epsilon : epsilons) {
                System.out.println("##################### Metric "+ metricName+" epsilon " + epsilon);
                for (int i = 0; i < iterations; i++) {
                    String filePath = outputPath + "/"+metricName+"_eps-" + epsilon + "_" + (i + 1) + ".csv";
                    Data data = initData(dataset, attributes);
                    ARXAnonymizer anonymizer = new ARXAnonymizer();
                    EDDifferentialPrivacy criterion = new EDDifferentialPrivacy(epsilon, delta);
                    ARXConfiguration config = ARXConfiguration.create();
                    config.setQualityModel(metrics.get(metricName));
                    config.addPrivacyModel(criterion);
                    config.setSuppressionLimit(1d);  //Not sure if this need to be specified or how it affects the result
                    config.setHeuristicSearchStepLimit(steps);
                    config.setHeuristicSearchStepSemantics(ARXConfiguration.SearchStepSemantics.EXPANSIONS);
                    if (epsilon <= 0.1) {
                        config.setDPSearchBudget(epsilon/10);
                    }
                    ARXResult result = anonymizer.anonymize(data, config);
                    DataHandle optimal = result.getOutput();
                    saveToFile(filePath, optimal);
                    System.out.println(i + " done out of " + iterations);
                }
            }
        }
    }

}
