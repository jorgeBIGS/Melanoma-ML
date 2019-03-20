package minerva.melanoma.classifiers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;

public class SimpleWrapImpl implements Classifier, Serializable, AutoencoderClassifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private List<Classifier> cls;

	private Normalize normalize;
	private SMOTE smote;
	private AttributeSelection selection;

	public SimpleWrapImpl(Classifier cl) {
		this.cls = new ArrayList<>();
		cls.add(cl);
		try {
			cls.add(cl.getClass().newInstance());
		} catch (Exception e) {
			throw new IllegalArgumentException();
		}
	}

	public void buildClassifier(Instances aux) throws Exception {
		Instances data = new Instances(generateData(aux));
		data.setRelationName("fusionado");
		data.setClassIndex(data.numAttributes() - 1);
		cls.get(0).buildClassifier(aux);
		cls.get(1).buildClassifier(data);
	}

	public double classifyInstance(Instance instance) throws Exception {
		return Utils.maxIndex(distributionForInstance(instance));
	}

	public double[] distributionForInstance(Instance aux) throws Exception {
		Instances test = new Instances(aux.dataset(), 0);
		Instance clone = new DenseInstance(aux);
		test.add(clone);
		clone.setDataset(test);

		Instances auxiliar = updateData(test);
		auxiliar.setRelationName("Test");
		auxiliar.setClassIndex(auxiliar.numAttributes() - 1);
		double[] origen = cls.get(0).distributionForInstance(aux);
		double[] preprocessed = cls.get(1).distributionForInstance(auxiliar.firstInstance());
		return aux.classAttribute().isNominal() ? combine(origen, preprocessed) : average(origen, preprocessed);

	}

	private double[] average(double[] origen, double[] autoencoded) {
		double[] result = new double[origen.length];
		for (int i = 0; i < origen.length; i++) {
			result[i] = (origen[i] + autoencoded[i]) / 2;
		}
		return result;
	}

	private double[] combine(double[] origen, double[] autoencoded) {
		double[] result = new double[origen.length];
		for (int i = 0; i < origen.length; i++) {
			result[i] = origen[i] + autoencoded[i];
		}
		Utils.normalize(result);
		return result;
	}

	private Instances updateData(Instances testing) throws Exception {

		Instances test = new Instances(testing);
		test = Filter.useFilter(test, normalize);
		if (testing.classAttribute().isNominal()) {
			test = Filter.useFilter(test, smote);
		}

		return Filter.useFilter(test, selection);
	}

	public Capabilities getCapabilities() {
		return cls.get(0).getCapabilities();
	}

	public Instances generateData(Instances aux) throws Exception {

		Instances data = new Instances(aux);

		if (aux.classAttribute().isNominal()) {
			smote = new SMOTE();
			int[] counts = aux.attributeStats(aux.classIndex()).nominalCounts;
			double ratio = 100 * 100 / (100.0 * counts[Utils.minIndex(counts)] / counts[Utils.maxIndex(counts)]);
			smote.setPercentage(ratio);
			smote.setInputFormat(data);
			smote.setRandomSeed(1);
			data = Filter.useFilter(data, smote);
		}

		// TODO: antes de SMOTE?
		normalize = new Normalize();
		normalize.setInputFormat(aux);
		data = Filter.useFilter(aux, normalize);

		data = getSelectedDataFin(data);

		return data;

	}

	private Instances getSelectedDataFin(Instances aux) throws Exception {

		selection = new AttributeSelection();
		selection.setInputFormat(aux);

		return Filter.useFilter(aux, selection);
	}
}
