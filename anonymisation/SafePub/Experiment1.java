package experiments;
import org.deidentifier.arx.*;
import org.deidentifier.arx.criteria.EDDifferentialPrivacy;
import org.deidentifier.arx.metric.Metric;

import java.io.IOException;
import java.util.HashMap;


public class Experiment1 extends Experiment {

    public static void main(String[] args) throws IOException {
        String[] attributes = new String[]{"age_years", "body_pain","body_temperature","body_weight_kg",
                "cold_and_flu_symptoms", "cold_and_flu_symptoms_duration", "medicine_allergy", "sex", "take_medicines",
                "skin_rash_symptom","works_with_children_elderly_or_sick_people"};

        //String[] attributes = new String[]{"longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
        //        "population","households","median_income","median_house_value"};

        //String[] attributes = new String[20];
        //for (int i = 1; i <= 20; i++) {
        //    attributes[i-1] = Integer.toString(i);
        //}

        double delta = 1e-5;

        run("caf", attributes, delta);
    }

    public static void run(String dataset, String[] attributes, double delta) throws IOException {

        int iterations = 50;
        int steps = 300;
        String outputPath = ""; // TODO: add output path
        HashMap<String, Metric> metrics = new HashMap<>();
        metrics.put("granularity", Metric.createLossMetric());
        metrics.put("intensity", Metric.createPrecisionMetric(true));
        metrics.put("discernibility", Metric.createDiscernabilityMetric(true));
        metrics.put("entropy", Metric.createEntropyMetric(true));
        metrics.put("groupsize",  Metric.createAECSMetric());

        double[] epsilons = {2d, 1.5, 1.25, Math.log(3), 1d, 0.75, Math.log(2), 0.5, 0.1, 0.01};

        System.out.println("--------------------------------- Running test 1.1");

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

