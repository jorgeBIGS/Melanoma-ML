package minerva.supervised.learners;

import java.util.ArrayList;
import java.util.List;

import es.us.indices.KMeansIndices;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.MLPAutoencoder;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ClusterizedAlgorithm extends AbstractClassifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7300188244861417901L;

	private static final int OFFSET = 2;

	private Integer MAX_CLUSTERS = 20;

	private NominalToBinary filter0;

	private ReplaceMissingValues filter1;

	private Normalize filter2;

	private MLPAutoencoder filter3;

	private AttributeSelection filter4;

	private List<SimpleKMeans> kmeansObject = new ArrayList<>();

	private Instances dataset;

	private Classifier classifier;

	private Integer selected;

	private int hiddenNeurons;

	public ClusterizedAlgorithm(Integer tag, Integer hidden) {
		classifier = new J48();
		hiddenNeurons = hidden;
		selected = tag;
	}

	public ClusterizedAlgorithm(Integer tag, Integer hidden, Classifier cl) {
		this(tag, hidden);
		classifier = cl;
	}

	public void buildClassifier(Instances data) throws Exception {

		Instances optimized = optimize(data);

		NumericToNominal nominal = new NumericToNominal();
		int[] indexes = new int[kmeansObject.size()];
		for (int i = 0; i < indexes.length; i++) {
			indexes[i] = optimized.numAttributes() - 1 - i;
		}
		nominal.setAttributeIndicesArray(indexes);

		nominal.setInputFormat(optimized);

		dataset = Filter.useFilter(optimized, nominal);

		classifier.buildClassifier(dataset);

	}

	public double[] distributionForInstance(Instance aux) throws Exception {
		double[] result = new double[dataset.numClasses()];
		try {
			int val = (int) classifyInstance(aux);
			result[val] = 1;
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(aux);
		}
		return result;
	}

	public double classifyInstance(Instance instance) throws Exception {
		filter0.input(instance);
		Instance i0 = filter0.output();
		filter1.input(i0);
		Instance i1 = filter1.output();
		filter2.input(i1);
		Instance i2 = filter2.output();
		filter3.input(i2);
		Instance i3 = filter3.output();
		filter4.input(i3);
		Instance i4 = filter4.output();

		i4.setDataset(null);
		i4.deleteAttributeAt(i4.numAttributes() - 1);

		instance.setDataset(null);
		for (SimpleKMeans kmeans : kmeansObject) {
			instance.insertAttributeAt(instance.numAttributes());
			Integer cluster = kmeans.clusterInstance(i4);
			instance.setValue(instance.numAttributes() - 1, cluster);
		}
		instance.setDataset(dataset);

		return classifier.classifyInstance(instance);
	}

	private Instances optimize(Instances data) throws Exception {
		Instances result = new Instances(data);

		Instances auxiliar = preprocessData(data);

		auxiliar.setClassIndex(-1);
		auxiliar.deleteAttributeType(Attribute.NOMINAL);
		auxiliar.deleteAttributeType(Attribute.DATE);
		auxiliar.deleteAttributeType(Attribute.STRING);
		auxiliar.deleteAttributeType(Attribute.RELATIONAL);

		List<Integer> bestIndexes = getIndexes(auxiliar);

		for (int i = 0; i < bestIndexes.size(); i++) {
			Attribute atrib = new Attribute("C" + i);
			result.insertAttributeAt(atrib, result.numAttributes());
			SimpleKMeans kmeans = new SimpleKMeans();
			kmeans.setNumClusters(bestIndexes.get(i));
			kmeans.setPreserveInstancesOrder(true);
			kmeans.buildClusterer(auxiliar);
			kmeansObject.add(kmeans);
			int[] assignments = kmeans.getAssignments();
			for (int j = 0; j < auxiliar.numInstances(); j++) {
				result.instance(j).setValue(result.numAttributes() - 1, assignments[j]);
			}

		}

		return result;
	}

	private List<Integer> getIndexes(Instances aux) throws Exception {
		List<Integer> result = new ArrayList<>();
		double[] bouldin = new double[MAX_CLUSTERS - 1];
		double[] dunn = new double[MAX_CLUSTERS - 1];
		double[] silhouette = new double[MAX_CLUSTERS - 1];
		double[] calinski = new double[MAX_CLUSTERS - 1];
		for (int i = 2; i < MAX_CLUSTERS; i++) {
			KMeansIndices kmeans = new KMeansIndices(i);
			kmeans.setPreserveInstancesOrder(true);
			kmeans.setNumClusters(i);
			kmeans.setInitializationMethod(new SelectedTag(selected, SimpleKMeans.TAGS_SELECTION));
			kmeans.buildClusterer(aux);
			kmeans.generateStructure(aux);
			kmeans.calculaIndices();
			silhouette[i - 1] = kmeans.getSilhouette().getResultado();
			dunn[i - 1] = kmeans.getDunn().getResultado();
			bouldin[i - 1] = kmeans.getDavidBouldin().getResultado();

			calinski[i - 1] = kmeans.getCalinskiHarabasz().getResultado();
		}

		result.add(Utils.maxIndex(silhouette) + OFFSET);
		result.add(Utils.maxIndex(dunn) + OFFSET);
		result.add(Utils.maxIndex(bouldin) + OFFSET);
		result.add(Utils.maxIndex(calinski) + OFFSET);

		return result;
	}

	private Instances preprocessData(Instances todos) throws Exception {
		Instances newInstances = new Instances(todos);

		newInstances.setClassIndex(0);

		filter0 = new NominalToBinary();
		filter0.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter0);

		filter1 = new ReplaceMissingValues();
		filter1.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter1);

		filter2 = new Normalize();
		filter2.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter2);

		filter3 = new MLPAutoencoder();
		filter3.setFilterType(new SelectedTag(MLPAutoencoder.FILTER_NONE, MLPAutoencoder.TAGS_FILTER));
		filter3.setNumFunctions(hiddenNeurons);
		filter3.setUseContractiveAutoencoder(true);
		filter3.setOutputInOriginalSpace(true);
		filter3.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter3);

		filter4 = new AttributeSelection();
		WrapperSubsetEval eval = new WrapperSubsetEval();
		eval.setClassifier(classifier.getClass().getConstructor().newInstance());
		filter4.setEvaluator(eval);
		filter4.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter4);

		return newInstances;
	}

	@Override
	public String toString() {
		return classifier.toString();
	}

}
