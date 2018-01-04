import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

class RRImage {
    int label;
    int predicted_label;
    ArrayList<Double> data;
    RRImage(){
        data = new ArrayList<>();
        predicted_label = -1;
    }
}

class RRClassifierThread implements Runnable {

    int lamda_index;
    double lamda[];
    ArrayList<RRImage> test_set; // N * M
    double[][][] weights;
    double[] accuracyArray;
    int[][] output;

    public RRClassifierThread(int lamda_index, double lamda[], ArrayList<RRImage> test_set, double[][][] weights, double[] accuracyArray, int[][] output){
        this.lamda_index = lamda_index;
        this.lamda = lamda;
        this.test_set = test_set;
        this.weights = weights;
        this.accuracyArray = accuracyArray;
        this.output = output;
    }

    public int classify(ArrayList<Double> data) {

        int max_index = -1;
        double max_value = -Double.MAX_VALUE;
        int num_attributes = data.size();
        for(int weight_index = 0; weight_index < 10; weight_index++) {
            double dot_pdt = 0;
            for(int j = 0; j < num_attributes; j++) {
                dot_pdt += data.get(j) * weights[lamda_index][weight_index][j];
            }
            if(dot_pdt > max_value) {
                max_value = dot_pdt;
                max_index = weight_index;
            }
        }

        return max_index;
    }

    public void findAccuracy() {
        int correct_predictions = 0;
        for(int i = 0; i < test_set.size(); i++) {
            RRImage test_image = test_set.get(i);
            test_image.predicted_label = classify(test_image.data);
            output[lamda_index][i] = test_image.predicted_label;
            if(test_image.predicted_label == test_image.label) {
                correct_predictions++;
            }
        }
        accuracyArray[lamda_index] = ((double)correct_predictions / (double)test_set.size());
    }

    @Override
    public void run() {
        findAccuracy();
    }

}

class RRWeightUpdaterThread implements Runnable {

    double positive_class = 1.0;
    double negative_class = -1.0;
    int lamda_index;
    double lamda[];
    int classifier;
    ArrayList<RRImage> train_set; // N * M
    double[][][] weights;

    public RRWeightUpdaterThread(int lamda_index, double lamda[], int classifier, ArrayList<RRImage> train_set, double[][][] weights){
        this.lamda_index = lamda_index;
        this.lamda = lamda;
        this.classifier = classifier;
        this.train_set = train_set;
        this.weights = weights;
    }

    public void updateWeightForAttribute(int attr_index) {

        double numerator = 0;
        double denominator_part1 = 0;


        int num_attributes = train_set.get(0).data.size();
        for(RRImage img : train_set) {

            double value_at_attr_index = img.data.get(attr_index);
            denominator_part1 += value_at_attr_index * value_at_attr_index;

            double inner_dot_pdt = 0.0;
            for(int j = 0; j < num_attributes; j++) {
                if(j != attr_index) {
                    inner_dot_pdt += img.data.get(j) * weights[lamda_index][classifier][j];
                }
            }

            double subtracted_val = 0.0;
            if(img.label == classifier) {
                subtracted_val = positive_class - inner_dot_pdt;
            } else {
                subtracted_val = negative_class - inner_dot_pdt;
            }

            numerator += subtracted_val * value_at_attr_index;
        }

        double denominator = denominator_part1 + lamda[lamda_index];

        weights[lamda_index][classifier][attr_index] =  numerator / denominator;
    }

    public double computeObjectiveFunctionValue() {
        double result = 0;

        double errors_norm_squared = 0;
        int num_attributes = train_set.get(0).data.size();
        for(RRImage img : train_set) {
            double inner_dot_pdt = 0;
            for(int j = 0; j < num_attributes; j++) {
                inner_dot_pdt += img.data.get(j) * weights[lamda_index][classifier][j];
            }

            double subtracted_val = 0;
            if(img.label == classifier) {
                subtracted_val = inner_dot_pdt - positive_class;
            } else {
                subtracted_val = inner_dot_pdt - negative_class;
            }

            errors_norm_squared += Math.pow(subtracted_val,2);
        }

        double weights_norm_squared = 0;
        for(int j = 0; j < num_attributes; j++) {
            weights_norm_squared += Math.pow(weights[lamda_index][classifier][j],2);
        }

        result = errors_norm_squared + lamda[lamda_index] * weights_norm_squared;
        return result;
    }

    public void performRidgeRegression() {

        int num_attributes = train_set.get(0).data.size();

        // create model
        double prev_obj_value = computeObjectiveFunctionValue();
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
                    updateWeightForAttribute(random_attr);
                    //System.out.println("Classifier : "+ weight_index+" Updated weight for : "+random_attr);
                }
            }

            double new_obj_value = computeObjectiveFunctionValue();

            iter++;
            long end = System.currentTimeMillis();
            //System.out.println("Classifier : "+ classifier+" Iteration Completed : "+iter+" prev_obj_value : "+prev_obj_value+" new_obj_value : "+new_obj_value+" value : "+(prev_obj_value - new_obj_value) / prev_obj_value+" time : "+(end - start)/1000+" secs");
            if((prev_obj_value - new_obj_value) / prev_obj_value < 0.001) {
                break;
            }
            prev_obj_value = new_obj_value;
        }
    }

    @Override
    public void run() {
        performRidgeRegression();
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
        } else {
            System.out.println("Wrong number of arguments");
            //return;
        }
        //read files
        ArrayList<RRImage> train_set = readCSV(train_file_path);
        ArrayList<RRImage> validation_set = readCSV(validation_file_path);
        ArrayList<RRImage> test_set = readCSV(test_file_path);

        System.out.println("completed reading files");

        long start = System.currentTimeMillis();

        int num_attributes = train_set.get(0).data.size();

        double[] lamdas = {0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0};

        double[][][] weights = new double[lamdas.length][10][num_attributes];

        int[][] output = new int[lamdas.length][validation_set.size()];

        for(int lamda_index = 0; lamda_index < lamdas.length; lamda_index++) {
            for(int classifier_index = 0; classifier_index < 10; classifier_index++) {
                Arrays.fill(weights[lamda_index][classifier_index], 0.0);
            }
        }

        double[] accuracy_array = findBestLamda(train_set,validation_set,lamdas, weights, output);

        double max_lamda = lamdas[0];
        double max_acc = Double.MIN_VALUE;

        for(int i = 0; i < lamdas.length; i++) {
            System.out.println("Lamda : "+lamdas[i]+" accuracy : "+ accuracy_array[i]);
            if(accuracy_array[i] > max_acc) {
                max_acc = accuracy_array[i];
                max_lamda = lamdas[i];
            }
        }

        System.out.println("Selected lamda : "+max_lamda);

        double[] selected_lamdas = {max_lamda, 2.0*max_lamda};

        ArrayList<RRImage> new_train_data = new ArrayList<>(train_set);
        new_train_data.addAll(validation_set);

        weights = new double[selected_lamdas.length][10][num_attributes];
        output = new int[selected_lamdas.length][test_set.size()];

        for(int lamda_index = 0; lamda_index < selected_lamdas.length; lamda_index++) {
            for(int classifier_index = 0; classifier_index < 10; classifier_index++) {
                Arrays.fill(weights[lamda_index][classifier_index], 0.0);
            }
        }

        accuracy_array = findBestLamda(new_train_data,test_set,selected_lamdas,weights, output);

        int max_lamda_index = 0;
        max_acc = Double.MIN_VALUE;

        for(int i = 0; i < selected_lamdas.length; i++) {
            //System.out.println("Lamda : "+selected_lamdas[i]+" accuracy : "+ accuracy_array[i]);
            if(accuracy_array[i] > max_acc) {
                max_acc = accuracy_array[i];
                max_lamda_index = i;
            }
        }

        System.out.println("ACCURACY : "+max_acc);

        writeOutputFile(output_file_path,output,max_lamda_index);
        writeWeightsFile(weights_file_path,weights,max_lamda_index);

        long end = System.currentTimeMillis();
        System.out.println("total time taken : "+(end - start)/1000 +" secs");

    }

    public static void writeOutputFile(String output_file_path, int[][] output, int lamda_index) throws IOException {
        BufferedWriter output_file_writer = new BufferedWriter(new FileWriter(output_file_path));
        for(int val :output[lamda_index]) {
            output_file_writer.write(val+"");
            output_file_writer.newLine();
        }
        output_file_writer.close();
    }

    public static void writeWeightsFile(String weights_file_path, double[][][] weights, int lamda_index) throws IOException {
        BufferedWriter weights_file_writer = new BufferedWriter(new FileWriter(weights_file_path));
        for(double[] class_weight : weights[lamda_index]) {
            for(double weight: class_weight) {
                weights_file_writer.write(weight+",");
            }
            weights_file_writer.newLine();
        }
        weights_file_writer.close();
    }

    public static double[] findBestLamda(ArrayList<RRImage> train_set, ArrayList<RRImage> test_set, double lamdas[], double[][][] weights, int[][] output) {

        ExecutorService executor = Executors.newCachedThreadPool();

        for(int lamda_index = 0; lamda_index < lamdas.length; lamda_index++) {
            for(int classifier_index = 0; classifier_index < 10; classifier_index++) {
                Runnable worker = new RRWeightUpdaterThread(lamda_index,lamdas,classifier_index,train_set,weights);
                executor.execute(worker);
            }
        }

        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        //System.out.println("Finished computing weights");

        double[] accuracy_array = new double[lamdas.length];
        Arrays.fill(accuracy_array, 0.0);

        executor = Executors.newCachedThreadPool();

        for(int lamda_index = 0; lamda_index < lamdas.length; lamda_index++) {
            Runnable worker = new RRClassifierThread(lamda_index,lamdas,test_set,weights,accuracy_array, output);
            executor.execute(worker);
        }

        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        //System.out.println("Finished computing accuracy");

        return accuracy_array;
    };

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
