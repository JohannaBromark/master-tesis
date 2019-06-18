package experiments;

import org.deidentifier.arx.*;
import org.deidentifier.arx.criteria.EDDifferentialPrivacy;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.metric.Metric;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Iterator;

/**
 * k-5
 * */
public class ExperimentK extends Experiment {

    public static void main(String[] args) throws IOException {

        //String[] attributes = new String[]{"age_years", "body_pain","body_temperature","body_weight_kg",
        //        "cold_and_flu_symptoms", "cold_and_flu_symptoms_duration", "medicine_allergy", "sex", "take_medicines",
        //        "skin_rash_symptom","works_with_children_elderly_or_sick_people"};

        String[] attributes = new String[]{"longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                  "population", "households", "median_income", "median_house_value"};

        String testNum = "3_1";

        run("housing_small", attributes, true, testNum);
        //run("caf", attributes, false);


    }

    public static void run(String dataset, String[] attributes, boolean allowSuppression, String testNum) throws IOException {
        String outputPath = ""; //TODO: add output path
        Data data = initData(dataset, attributes);
        double suppressionLimit = allowSuppression? 1d : 0d;
        String outputFileName = allowSuppression? "k5_suppression.csv": "k5_no_suppression.csv";

        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXConfiguration config = ARXConfiguration.create();
        config.addPrivacyModel(new KAnonymity(5));
        config.setSuppressionLimit(suppressionLimit);
        config.setQualityModel(Metric.createLossMetric());

        ARXResult result = anonymizer.anonymize(data, config);

        String filePath = outputPath +"/"+ outputFileName;

        DataHandle optimal = result.getOutput();
        saveToFile(filePath, optimal);

    }

}
