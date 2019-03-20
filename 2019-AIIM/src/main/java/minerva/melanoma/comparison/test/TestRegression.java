package minerva.melanoma.comparison.test;

import java.io.File;
import java.io.FileWriter;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import minerva.melanoma.comparison.TesterReg;

public class TestRegression {

	private static final File DATA_DIR = new File("data");


	public static void main(String... args) throws Exception {
		File dir = new File("results");
		dir.mkdirs();
		FileWriter result = new FileWriter("results/regression.csv");
		result.write("Dataset/Algor, Original, Preprocessed Original, Parallel, Cascade, Both" + "\n");
		
		ExecutorService executor = Executors.newFixedThreadPool(5);
		for (File f : DATA_DIR.listFiles()) {
			if (!f.isDirectory()) {
				Runnable worker = new TesterReg(f, 100., result);
				executor.execute(worker);
				worker = new TesterReg(f, 75., result);
				executor.execute(worker);
				worker = new TesterReg(f, 50.0, result);
				executor.execute(worker);
				worker = new TesterReg(f, 25.0, result);
				executor.execute(worker);
			}
		}
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");
        result.close();
		
	}

	

}
