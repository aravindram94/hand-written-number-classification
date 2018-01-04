import java.io.*;
import java.util.*;

class RRImage {
    int label;
    int predicted_label;
    ArrayList<Double> data;
    RRImage(){
        data = new ArrayList<>();
        predicted_label = -1;
    }
}

class RidgeRegression{
    // N is the number of records
    //M is the number of attributes
    ArrayList<RRImage> train_set; // N * M
    ArrayList<RRImage> test_set; // N * M
    ArrayList<ArrayList<Double>> weights; // 10 * 1M
    double lamda;
    String output_file_path;
    String weights_file_path;
    double positive_class = 1.0;
    double negative_class = -1.0;
    RidgeRegression(ArrayList<RRImage> train_set, ArrayList<RRImage> test_set, double lamda, String output_file_path, String weights_file_path) {
        this.train_set = train_set;
        this.test_set = test_set;
        this.lamda = lamda;
        this.output_file_path = output_file_path;
        this.weights_file_path = weights_file_path;
        initializeWeights();
    }
    public void initializeWeights() {
        weights = new ArrayList<>();
        int attr_size = train_set.get(0).data.size();
        Double[] initial_weights = new Double[attr_size]; // 10 is the number of labels -> 0 to 9
        Arrays.fill(initial_weights,0.0);
        for(int i = 0; i < 10; i++) {
            weights.add(new ArrayList<>(Arrays.asList(initial_weights)));
        }
    }

    public void updateWeightForAttribute(int weight_index, int attr_index, double lamda) {

        double numerator = 0;
        double denominator_part1 = 0;


        int num_attributes = train_set.get(0).data.size();
        for(RRImage img : train_set){

            double value_at_attr_index = img.data.get(attr_index);
            denominator_part1 += value_at_attr_index * value_at_attr_index;

            double inner_dot_pdt = 0.0;
            for(int j = 0; j < num_attributes; j++) {
                if(j != attr_index) {
                    inner_dot_pdt += img.data.get(j) * weights.get(weight_index).get(j);
                }
            }

            double subtracted_val = 0.0;
            if(img.label == weight_index) {
                subtracted_val = positive_class - inner_dot_pdt;
            } else {
                subtracted_val = negative_class - inner_dot_pdt;
            }

            numerator += subtracted_val * value_at_attr_index;
        }

        double denominator = denominator_part1 + lamda;

        weights.get(weight_index).set(attr_index, numerator / denominator);
    }

    public double computeObjectiveFunctionValue(int weight_index, double lamda) {
        double result = 0;

        double errors_norm_squared = 0;
        int num_attributes = train_set.get(0).data.size();
        for(RRImage img : train_set) {
            double inner_dot_pdt = 0;
            for(int j = 0; j < num_attributes; j++) {
                inner_dot_pdt += img.data.get(j) * weights.get(weight_index).get(j);
            }

            double subtracted_val = 0;
            if(img.label == weight_index) {
                subtracted_val = inner_dot_pdt - positive_class;
            } else {
                subtracted_val = inner_dot_pdt - negative_class;
            }

            errors_norm_squared += Math.pow(subtracted_val,2);
        }

        double weights_norm_squared = 0;
        for(int j = 0; j < num_attributes; j++) {
            weights_norm_squared += Math.pow(weights.get(weight_index).get(j),2);
        }

        result = errors_norm_squared + lamda * weights_norm_squared;
        return result;
    }

    public int classify(ArrayList<Double> data) {

        int max_index = -1;
        double max_value = -Double.MAX_VALUE;
        int num_attributes = data.size();
        for(int weight_index = 0; weight_index < 10; weight_index++) {
            double dot_pdt = 0;
            for(int j = 0; j < num_attributes; j++) {
                dot_pdt += data.get(j) * weights.get(weight_index).get(j);
            }
            if(dot_pdt > max_value) {
                max_value = dot_pdt;
                max_index = weight_index;
            }
        }

        return max_index;
    }

    public double performRidgeRegression() throws IOException {

        int num_attributes = train_set.get(0).data.size();

        // create model
        for(int weight_index = 0; weight_index < 10; weight_index++) {
            double prev_obj_value = computeObjectiveFunctionValue(weight_index,lamda);
            int iter = 0;
            while(true) {
                long start = System.currentTimeMillis();
                Random rand = new Random();
                HashSet<Integer> processed_attributes = new HashSet<>();
                int processed_count = 0;
                while(num_attributes != processed_count) {
                    //generate random attribute
                    int random_attr = rand.nextInt(num_attributes);
                    if(processed_attributes.add(random_attr)) {
                        processed_count++;
                        updateWeightForAttribute(weight_index, random_attr, lamda);
                    }
                }

                double new_obj_value = computeObjectiveFunctionValue(weight_index,lamda);

                iter++;
                long end = System.currentTimeMillis();
                //System.out.println("Classifier : "+ weight_index+" Iteration Completed : "+iter+" prev_obj_value : "+prev_obj_value+" new_obj_value : "+new_obj_value+" value : "+(prev_obj_value - new_obj_value) / prev_obj_value+" time : "+(end - start)/1000+" secs");
                if((prev_obj_value - new_obj_value) / prev_obj_value < 0.0001) {
                    break;
                }
                prev_obj_value = new_obj_value;
            }
        }

        //classify using the above model
        int correct_predictions = 0;
        for(RRImage test_image:test_set) {
            test_image.predicted_label = classify(test_image.data);
            if(test_image.predicted_label == test_image.label) {
                correct_predictions++;
            }
        }

        return ((double)correct_predictions / (double)test_set.size());
    }

    public void writeOutputFile() throws IOException {
        BufferedWriter output_file_writer = new BufferedWriter(new FileWriter(output_file_path));
        for(RRImage test_image:test_set) {
            test_image.predicted_label = classify(test_image.data);
            output_file_writer.write(test_image.predicted_label+"");
            output_file_writer.newLine();
        }
        output_file_writer.close();
    }

    public void writeWeightsFile() throws IOException {
        BufferedWriter weights_file_writer = new BufferedWriter(new FileWriter(weights_file_path));
        for(ArrayList<Double> class_weight : weights) {
            for(double weight: class_weight) {
                weights_file_writer.write(weight+",");
            }
            weights_file_writer.newLine();
        }
        weights_file_writer.close();
    }

}



public class Main {

    public static void main(String[] args) throws IOException {

        String train_file_path = "../rep2/mnist_train.csv";
        String validation_file_path = "../rep2/mnist_validation.csv";
        String test_file_path = "../rep2/mnist_test.csv";
        String output_file_path = "output.txt";
        String weights_file_path = "weights_file.csv";

        if(args.length == 5) {
            train_file_path = args[0];
            validation_file_path = args[1];
            test_file_path = args[2];
            output_file_path = args[3];
            weights_file_path = args[4];
        }
        //read files
        ArrayList<RRImage> train_set = readCSV(train_file_path);
        ArrayList<RRImage> validation_set = readCSV(validation_file_path);
        ArrayList<RRImage> test_set = readCSV(test_file_path);

        System.out.println("completed reading files");

        long start = System.currentTimeMillis();

        double[] lamdas = {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0};
        double max_lamda = lamdas[0];
        double max_acc = Double.MIN_VALUE;
        for(int i = 0; i < lamdas.length; i++) {
            System.out.println("Evaluating for lamda : "+lamdas[i]);
            long starttime = System.currentTimeMillis();
            RidgeRegression rr_classifier = new RidgeRegression(train_set,validation_set,lamdas[i],output_file_path,weights_file_path);
            double acc = rr_classifier.performRidgeRegression();
            System.out.println("EVALUATION ACCURACY : "+acc+" LAMDA : "+lamdas[i]+" time taken : "+(System.currentTimeMillis() - starttime) / 1000 +" secs");
            if(max_acc < acc) {
                max_acc = acc;
                max_lamda = lamdas[i];
            }
        }
        System.out.println("Selected lamda : "+max_lamda);

        ArrayList<RRImage> new_train_data = new ArrayList<>(train_set);
        new_train_data.addAll(validation_set);

        // Test using selected lamda
        System.out.println("EVALUATING USING SELECTED LAMDA : "+max_lamda);
        long starttime = System.currentTimeMillis();
        RidgeRegression rr_classifier1 = new RidgeRegression(new_train_data,test_set,max_lamda,output_file_path,weights_file_path);
        double acc1 = rr_classifier1.performRidgeRegression();
        //System.out.println("TESTING ACCURACY : "+acc1+" LAMDA : "+max_lamda+" time taken : "+(System.currentTimeMillis() - starttime) / 1000 +" secs");

        // Test using selected lamda * 2
        System.out.println("EVALUATING USING SELECTED LAMDA * 2 : "+max_lamda * 2.0);
        starttime = System.currentTimeMillis();
        RidgeRegression rr_classifier2 = new RidgeRegression(new_train_data,test_set,max_lamda * 2.0,output_file_path,weights_file_path);
        double acc2 = rr_classifier2.performRidgeRegression();
        //System.out.println("TESTING ACCURACY : "+acc2+" LAMDA : "+max_lamda * 2.0+" time taken : "+(System.currentTimeMillis() - starttime) / 1000 +" secs");

        if(acc1 > acc2) {
            System.out.println("Performance of lamda is better than 2 * lamda");
            System.out.println("ACCURACY : "+acc1 * 100.0);
            rr_classifier1.writeOutputFile();
            rr_classifier1.writeWeightsFile();
        } else {
            System.out.println("Performance of 2 * lamda is better than lamda");
            System.out.println("ACCURACY : "+acc2 * 100.0);
            rr_classifier2.writeOutputFile();
            rr_classifier2.writeWeightsFile();
        }

        long end = System.currentTimeMillis();
        System.out.println("total time taken : "+(end - start)/1000 +" secs");


    }

    public static ArrayList<RRImage> readCSV(String filename){
        File file= new File(filename);

        ArrayList<RRImage> csv_array = new ArrayList<>();
        Scanner inputStream;

        try{
            inputStream = new Scanner(file);

            while(inputStream.hasNext()) {
                String line= inputStream.next();
                RRImage img = new RRImage();
                String[] split = line.split(",");
                img.label = Integer.parseInt(split[0]);
                double sum_squared = 0;
                for(int i = 1; i < split.length; i++) {
                    sum_squared += Math.pow(Double.parseDouble(split[i]), 2);
                }
                sum_squared = Math.sqrt(sum_squared);
                for(int i = 1; i < split.length; i++) {
                    img.data.add(Double.parseDouble(split[i]) / sum_squared);
                }
                csv_array.add(img);
            }

            inputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return csv_array;
    }

}
