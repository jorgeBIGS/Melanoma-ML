package minerva.melanoma.test;

import java.io.File;
import java.util.Random;

import es.us.watchmaker.melanoma.IndividualAutoencoderImpl;
import es.us.watchmaker.melanoma.Preprocessor;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class TestExperimenterCV {

	private static final Integer FOLDS = 5;
	private static final Integer REPS = 10;
	private static final String DATA_PATH = "data/macarena.arff";

	public static void main(String[] args) throws Exception {
		ArffLoader csv = new ArffLoader();
		csv.setFile(new File(DATA_PATH));
	
		
		Instances todos = csv.getDataSet();
		todos.setClassIndex(0);
		todos.deleteAttributeAt(1);
		todos.deleteAttributeAt(1);
		
		Preprocessor processor = new Preprocessor(new int[] {0,1});
		todos = processor.preprocessData(todos, new IndividualAutoencoderImpl(3, new Random()));

		Evaluation eval4 = new Evaluation(todos);
		Classifier algor = new J48();
		for (int i = 0; i < REPS; i++) {

			
			eval4.crossValidateModel(algor, todos, FOLDS, new Random(i));

		}
		algor.buildClassifier(todos);
		System.out.println(algor);
		System.out.println(eval4.toSummaryString());
		System.out.println(eval4.toClassDetailsString());
		System.out.println(eval4.toMatrixString());

	}

	//
}
