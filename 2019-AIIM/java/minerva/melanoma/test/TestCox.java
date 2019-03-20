package minerva.melanoma.test;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import javastat.survival.regression.CoxRegression;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class TestCox {
	private static final String DATA_PATH = "data/melanomaClean/dataCleaned2.arff";
	// private static double ALPHA = 0.05;

	// private static final Double ALPHA = 0.05;

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		ArffLoader csv = new ArffLoader();
		csv.setFile(new File(DATA_PATH));
		Instances train = csv.getDataSet();
		
		Integer tIndex = 0;
		Integer cIndex = train.numAttributes()-1;

		System.out.println(train);

		showCoxStatistics(train, tIndex, cIndex);

		Instances done = train;//preprocess(train);

		System.out.println(done);

		showCoxStatistics(done, tIndex, cIndex);
	}

	private static Instances preprocess(Instances data) throws Exception {
		Instances newInstances = new Instances(data);
		newInstances.setClassIndex(0);

		// ReplaceMissingValues filter1 = new ReplaceMissingValues();
		// filter1.setInputFormat(newInstances);
		// newInstances = Filter.useFilter(newInstances, filter1);
		// newInstances.setClassIndex(0);

		Discretize filter2 = new Discretize();
		filter2.setAttributeIndicesArray(new int[] { 0, 1 });
		filter2.setInvertSelection(true);
		filter2.setMakeBinary(true);
		filter2.setBins(2);
		// filter2.setFindNumBins(true);
		filter2.setUseEqualFrequency(true);
		filter2.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter2);
		newInstances.setClassIndex(0);

//		NumericToNominal filter3 = new NumericToNominal();
//		filter3.setAttributeIndicesArray(new int[] { 0, 1, 2 });
//		filter3.setInvertSelection(true);
//		filter3.setInputFormat(newInstances);
//		newInstances = Filter.useFilter(newInstances, filter3);
//		newInstances.setClassIndex(0);

		// NominalToBinary filter4 = new NominalToBinary();
		// filter4.setBinaryAttributesNominal(false);
		// filter4.setInputFormat(newInstances);
		// newInstances = Filter.useFilter(newInstances, filter4);
		// newInstances.setClassIndex(0);

		// AddEvolutionaryKMeansCluster filter5 = new
		// AddEvolutionaryKMeansCluster("1,2");
		// newInstances = filter5.process(newInstances);

		// NominalToBinary filter6 = new NominalToBinary();
		// filter6.setInputFormat(newInstances);
		// newInstances = Filter.useFilter(newInstances, filter6);

		return newInstances;

	}

	private static void showCoxStatistics(Instances done, Integer tIndex, Integer cIndex) {

		double[] taim = done.attributeToDoubleArray(tIndex);
		double[] aim = done.attributeToDoubleArray(cIndex);

		double[][] attributes = new double[done.numAttributes()][done.numInstances()];

		for (int i = 0; i < done.numAttributes(); i++) {
			attributes[i] = done.attributeToDoubleArray(i);
		}

		double[] pValues = new double[done.numAttributes()];
		double[] exps = new double[pValues.length];

		for (int i = 0; i < done.numAttributes(); i++) {
			try {

				CoxRegression testclass1 = new CoxRegression(taim, aim, attributes[i]);
				pValues[i] = testclass1.pValue[0];
				exps[i] = Math.pow(Math.E, testclass1.coefficients[0]);
			} catch (RuntimeException e) {
				pValues[i] = Double.NaN;
				exps[i] = Double.NaN;
			}
		}

		Map<String, String> map = new HashMap<>();

		for (int i = 0; i < pValues.length; i++) {
			map.put(done.attribute(i).name(), "(" + pValues[i] + "," + exps[i] + ")");

			System.out.println(done.attribute(i).name() + " (" + pValues[i] + "," + exps[i] + ")");
		}
	}
}
