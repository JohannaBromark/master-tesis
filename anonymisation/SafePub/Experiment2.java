package experiments;

import org.deidentifier.arx.*;
import org.deidentifier.arx.criteria.EDDifferentialPrivacy;
import org.deidentifier.arx.metric.Metric;

import java.io.IOException;


/**
* Test 1.2 Vary delta, keep epsilon fixed. epsilon 1.0
* */
public class Experiment2 extends Experiment {

    public static void main(String[] args) throws IOException{

        //String[] attributes = new String[]{"age_years", "body_pain","body_temperature","body_weight_kg",
        //        "cold_and_flu_symptoms", "cold_and_flu_symptoms_duration", "medicine_allergy", "sex", "take_medicines",
        //        "skin_rash_symptom","works_with_children_elderly_or_sick_people"};
        //run("caf", attributes);

        String[] attributes = new String[]{"longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
                "population","households","median_income","median_house_value"};
        run("housing", attributes);
    }

    public static void run(String dataset, String[] attributes) throws IOException {
        int iterations = 50;
        int steps = 300;
        String outputPath = ""; //TODO: add outputpath
        double epsilon = 1d;
        double[] deltas = new double[]{1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16,
                1e-17, 1e-18, 1e-19, 1e-20};
        Metric metric = Metric.createLossMetric();

        System.out.println("--------------------------------- Running test 1.2");

        for (double delta : deltas) {
            System.out.println("##################### Delta "+ delta);
            for (int i = 0; i < iterations; i++) {
                String filePath = outputPath + "/delta_"+delta+"_"+(i+1)+".csv";
                Data data = initData(dataset, attributes);
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
                System.out.println(i + " done out of " + iterations);
            }
        }

    }

}


