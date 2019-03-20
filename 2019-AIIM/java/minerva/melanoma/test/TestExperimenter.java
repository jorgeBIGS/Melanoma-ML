package minerva.melanoma.test;

import java.io.File;

import minerva.supervised.learners.ClusterizedAlgorithm;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class TestExperimenter {

	// private static final Integer FOLDS = 5;
	// private static final Integer REPS = 10;

	public static void main(String[] args) throws Exception {
		CSVLoader loader = new CSVLoader();
		loader.setFile(new File("data/todos.csv"));

		Instances todos = loader.getDataSet();
		todos.setClassIndex(0);
		todos.deleteAttributeAt(1);
		todos.deleteAttributeAt(1);
		todos.deleteAttributeAt(1);

		Evaluation eval1 = new Evaluation(todos);
		Evaluation eval2 = new Evaluation(todos);
		Evaluation eval3 = new Evaluation(todos);
		Evaluation eval4 = new Evaluation(todos);

		Integer hidden = 8;
		Classifier algor = new ClusterizedAlgorithm(SimpleKMeans.RANDOM, hidden);
		algor.buildClassifier(todos);
		System.out.println(algor);
		eval1.evaluateModel(algor, todos);

		algor = new ClusterizedAlgorithm(SimpleKMeans.CANOPY, hidden);
		algor.buildClassifier(todos);
		System.out.println(algor);
		eval2.evaluateModel(algor, todos);

		algor = new ClusterizedAlgorithm(SimpleKMeans.FARTHEST_FIRST, hidden);
		algor.buildClassifier(todos);
		System.out.println(algor);
		eval3.evaluateModel(algor, todos);

		algor = new J48();
		algor.buildClassifier(todos);
		System.out.println(algor);
		eval4.evaluateModel(algor, todos);

		System.out.println(eval1.toSummaryString());
		System.out.println(eval1.toClassDetailsString());
		System.out.println(eval1.toMatrixString());
		System.out.println(eval2.toSummaryString());
		System.out.println(eval2.toClassDetailsString());
		System.out.println(eval2.toMatrixString());
		System.out.println(eval3.toSummaryString());
		System.out.println(eval3.toClassDetailsString());
		System.out.println(eval3.toMatrixString());
		System.out.println(eval4.toSummaryString());
		System.out.println(eval4.toClassDetailsString());
		System.out.println(eval4.toMatrixString());

	}

}
