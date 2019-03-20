package minerva.melanoma.comparison;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import minerva.melanoma.classifiers.AutoencoderType;
import minerva.melanoma.classifiers.AutoencoderWrapImpl;
import minerva.melanoma.classifiers.SimpleWrapImpl;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class Tester implements Runnable {

	private File file;
	private Double rate;
	private FileWriter result;
	private final Classifier[] classifiers = { new IBk(),new J48(), new NaiveBayes(), new MultilayerPerceptron(), 
			new IBk(3), new IBk(5), new IBk(7), new SMO(), new RandomForest() };
	private final int NUM_REPS = 10;
	private final int NUM_FOLDS = 10;

	public Tester(File file, Double rate, FileWriter result) {
		super();
		this.file = file;
		this.rate = rate;
		this.result = result;
	}

	public void run() {
		try {
			ArffLoader loader = new ArffLoader();

			loader.setFile(file);

			Instances aux = loader.getDataSet();
			aux.deleteAttributeAt(aux.numAttributes() - 2);
			aux.setClassIndex(aux.numAttributes() - 1);

			for (Classifier x : classifiers) {
				Evaluation eval = new Evaluation(aux);
				Evaluation evalIni = new Evaluation(aux);
				Evaluation evalBoth = new Evaluation(aux);
				Evaluation evalParallel = new Evaluation(aux);
				Evaluation evalCascade = new Evaluation(aux);

				for (int i = 1; i <= NUM_REPS; i++) {

					Resample sample = new Resample();
					sample.setRandomSeed(i);
					sample.setSampleSizePercent(rate);
					sample.setInputFormat(aux);
					Instances training = Filter.useFilter(aux, sample);
					
					try {
						evalIni.crossValidateModel(x, training, NUM_FOLDS, new Random(i));
						eval.crossValidateModel(new SimpleWrapImpl(x), training, NUM_FOLDS, new Random(i));
						evalParallel.crossValidateModel(new AutoencoderWrapImpl(x, AutoencoderType.PARALLEL), training,
								NUM_FOLDS, new Random(i));
						evalCascade.crossValidateModel(new AutoencoderWrapImpl(x, AutoencoderType.CASCADE), training,
								NUM_FOLDS, new Random(i));
						evalBoth.crossValidateModel(new AutoencoderWrapImpl(x, AutoencoderType.BOTH), training,
								NUM_FOLDS, new Random(i));
					} catch (Exception e) {
						e.printStackTrace();
					}

				}

				result.append(rate + "/" + file.getName() + "/" + x.getClass().getSimpleName()
						+ (x instanceof IBk ? ((IBk) x).getKNN() : "") + "," + evalIni.unweightedMacroFmeasure() + ","
						+ eval.unweightedMacroFmeasure() + "," + evalParallel.unweightedMacroFmeasure() + ","
						+ evalCascade.unweightedMacroFmeasure() + "," + evalBoth.unweightedMacroFmeasure() + "\n");

				System.out.println(rate + "/" + file.getName() + "/" + x.getClass().getSimpleName()
						+ (x instanceof IBk ? ((IBk) x).getKNN() : "") + "," + evalIni.unweightedMacroFmeasure() + ","
						+ eval.unweightedMacroFmeasure() + "," + evalParallel.unweightedMacroFmeasure() + ","
						+ evalCascade.unweightedMacroFmeasure() + "," + evalBoth.unweightedMacroFmeasure() + "\n");

			}
		} catch (Exception e1) {

			e1.printStackTrace();
		}
	}

}
