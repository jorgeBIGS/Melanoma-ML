package minerva.melanoma.test;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class TestSplitter {

	public static void main(String[] args) throws IOException {
		CSVLoader loader = new CSVLoader();
		loader.setFile(new File("data/todos.csv"));

		Instances todos = loader.getDataSet();

		split(todos);

	}

	private static void split(Instances preprocessed) throws IOException {
		Instances macarena = new Instances(preprocessed, 0, 218);
		Instances salamanca = new Instances(preprocessed, 218, preprocessed.numInstances() - 218);

		ArffSaver saver = new ArffSaver();
		saver.setFile(new File("data/macarena.arff"));
		saver.setInstances(macarena);
		saver.writeBatch();

		saver.setFile(new File("data/salamanca.arff"));
		saver.setInstances(salamanca);
		saver.writeBatch();

		saver.setFile(new File("data/todos.arff"));
		saver.setInstances(preprocessed);
		saver.writeBatch();
	}

}
