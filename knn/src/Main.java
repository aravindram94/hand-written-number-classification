import java.io.*;
import java.util.*;

class KNNImage {
    int label;
    int predicted_label;
    ArrayList<Double> data;
    KNNImage(){
        data = new ArrayList<>();
        predicted_label = -1;
    }
}

class Neighbor {
    KNNImage image;
    double similarity;
    Neighbor(KNNImage i) {
        image = i;
        similarity = 0;
    }
}

class NeighborComp implements Comparator<Neighbor> {
    @Override
    public int compare(Neighbor x, Neighbor y) {
        if (x.similarity > y.similarity){
            return 1;
        }
        if (x.similarity < y.similarity){
            return -1;
        }
        return 0;
    }
}

class KNearestNeighbours{
    ArrayList<KNNImage> train_set;
    ArrayList<KNNImage> test_set;
    ArrayList<KNNImage> validation_set;
    String output_file_path;
    KNearestNeighbours(ArrayList<KNNImage> train_set, ArrayList<KNNImage> test_set, ArrayList<KNNImage> validation_set, String output_file_path) {
        this.train_set = train_set;
        this.validation_set = validation_set;
        this.test_set = test_set;
        this.output_file_path = output_file_path;
    }
    public double getCosineSimilarity(KNNImage a, KNNImage b) {
        double cosine_dot_product = 0;
        for (int k = 0; k < a.data.size(); k++) {
            cosine_dot_product += a.data.get(k) * b.data.get(k);
        }
        return cosine_dot_product;
    }



    public ArrayList<Neighbor> getKNeighbors(int k, KNNImage test_image, ArrayList<KNNImage> train_data) {

        PriorityQueue<Neighbor> neighbors_heap = new PriorityQueue<>(k, new NeighborComp());

        for(int i = 0; i < k; i++) {
            Neighbor n = new Neighbor(train_data.get(i));
            n.similarity = getCosineSimilarity(test_image, train_data.get(i));
            neighbors_heap.add(n);
        }

        for(int i = k; i < train_data.size(); i++) {
            Neighbor n = new Neighbor(train_data.get(i));
            n.similarity = getCosineSimilarity(test_image, train_data.get(i));
            if(neighbors_heap.peek().similarity < n.similarity) {
                neighbors_heap.poll();
                neighbors_heap.add(n);
            }
        }
        ArrayList<Neighbor> neighbors = new ArrayList<>();
        Neighbor last;
        while ((last = neighbors_heap.poll()) != null) {
            neighbors.add(last);
        }
        return neighbors;
    }

    public int classify(ArrayList<Neighbor> neighbors, int k) {
        double[] labels = new double[10];
        Arrays.fill(labels, 0);
        for(int i = neighbors.size() - 1,j = 0; j < k; i--,j++) {
            labels[neighbors.get(i).image.label]++;
        }
        double max_val = -1;
        int max_index = -1;
        for(int i = 0; i < labels.length; i++) {
            if(labels[i] > max_val) {
                max_val = labels[i];
                max_index = i;
            }
        }
        return max_index;
    }

    public double accuracy(ArrayList<KNNImage> test_data) {
        double result = 0;
        int count = 0;
        int size = test_data.size();
        for(int i = 0; i < size; i++) {
            if(test_data.get(i).label == test_data.get(i).predicted_label) {
                count++;
            }
        }
        result = (double)count / (double)size;
        return result;
    }

    public int findK() {
        int best_k = -1;
        double best_accuracy = -1;

        ArrayList<ArrayList<Neighbor>> overall_neighbors_list = new ArrayList<>();
        for(int i = 0; i < validation_set.size(); i++) {
            overall_neighbors_list.add(getKNeighbors(20, validation_set.get(i), train_set));
        }
        for(int k = 1; k <=20; k++) {
            for(int i = 0; i < validation_set.size(); i++) {
                validation_set.get(i).predicted_label = classify(overall_neighbors_list.get(i),k);
            }
            double acc = accuracy(validation_set);
            if(best_accuracy < acc) {
                best_accuracy = acc;
                best_k = k;
            }
        }
        return best_k;
    }

    public void performKNN(ArrayList<KNNImage> train_data, ArrayList<KNNImage> test_data, int k) throws IOException {
        for(int i = 0; i < test_data.size(); i++) {
            ArrayList<Neighbor> neighbors = getKNeighbors(k, test_data.get(i), train_data);
            test_data.get(i).predicted_label = classify(neighbors, k);
        }
        double acc = accuracy(test_data);

        BufferedWriter output_file_writer = new BufferedWriter(new FileWriter(output_file_path));
        int size = test_data.size();
        for(int i = 0; i < size; i++) {
            output_file_writer.write(test_data.get(i).predicted_label+"");
            output_file_writer.newLine();
        }
        output_file_writer.close();
        System.out.println("ACCURACY : "+acc * 100.0);
    }
}

public class Main {

    public static void main(String[] args) throws IOException {

        String train_file_path = "../rep2/mnist_train.csv";
        String validation_file_path = "../rep2/mnist_validation.csv";
        String test_file_path = "../rep2/mnist_test.csv";
        String output_file_path = "output.txt";

        if(args.length == 4 || args.length == 5) {
            train_file_path = args[0];
            validation_file_path = args[1];
            test_file_path = args[2];
            output_file_path = args[3];
        } else {
            System.out.println("Wrong number of Arguments. Running default case");
        }
        //read files
        ArrayList<KNNImage> train_set = readCSV(train_file_path);
        ArrayList<KNNImage> validation_set = readCSV(validation_file_path);
        ArrayList<KNNImage> test_set = readCSV(test_file_path);

        System.out.println("completed reading files");

        long start = System.currentTimeMillis();

        //knn
        KNearestNeighbours knn_classifier = new KNearestNeighbours(train_set,test_set,validation_set,output_file_path);
        int k = knn_classifier.findK();
        System.out.println("Selected best k : "+k);
        ArrayList<KNNImage> new_train_data = new ArrayList<>(train_set);
        new_train_data.addAll(validation_set);
        knn_classifier.performKNN(new_train_data,test_set,k);


        long end = System.currentTimeMillis();
        System.out.println("time taken : "+(end - start)/1000 +" secs");


    }

    public static ArrayList<KNNImage> readCSV(String filename){
        File file= new File(filename);

        ArrayList<KNNImage> csv_array = new ArrayList<>();
        Scanner inputStream;

        try{
            inputStream = new Scanner(file);

            while(inputStream.hasNext()) {
                String line= inputStream.next();
                KNNImage img = new KNNImage();
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
