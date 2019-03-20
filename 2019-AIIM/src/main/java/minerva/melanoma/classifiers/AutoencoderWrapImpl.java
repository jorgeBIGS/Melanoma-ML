package minerva.melanoma.classifiers;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;

import minerva.weka.filters.unsupervised.attribute.MLPAutoencoder;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RenameAttribute;

public class AutoencoderWrapImpl implements Classifier, Serializable, AutoencoderClassifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private Classifier cl;
	private AutoencoderType type;
	private List<MLPAutoencoder> filtrosParallel;
	private List<MLPAutoencoder> filtrosCascade;

	private SMOTE smote;

	private AttributeSelection selectionIni;

	private AttributeSelection selectionMid;

	private AttributeSelection selectionFin;

	private Classifier original;

	private Normalize normalize;

	private static final Integer AUTOENCODER_FILTER = MLPAutoencoder.FILTER_NONE;

	public AutoencoderWrapImpl(Classifier cl, AutoencoderType type) {
		this.cl = cl;
		this.type = type;
		this.filtrosCascade = new LinkedList<MLPAutoencoder>();
		this.filtrosParallel = new LinkedList<MLPAutoencoder>();
	}

	public void buildClassifier(Instances aux) throws Exception {
		Instances data = new Instances(generateData(aux));
		data.setRelationName("fusionado");
		data.setClassIndex(data.numAttributes() - 1);
		original = cl.getClass().newInstance();
		original.buildClassifier(aux);
		cl.buildClassifier(data);
	}

	public Instances generateData(Instances aux) throws Exception {
		Instances dataAutoencoder = aux;

		// TODO: antes de SMOTE?
		normalize = new Normalize();
		normalize.setInputFormat(aux);
		dataAutoencoder = Filter.useFilter(dataAutoencoder, normalize);

		if (dataAutoencoder.classAttribute().isNominal()) {
			smote = new SMOTE();
			int[] counts = aux.attributeStats(aux.classIndex()).nominalCounts;
			double ratio = 100 * 100 / (100.0 * counts[Utils.minIndex(counts)] / counts[Utils.maxIndex(counts)]);
			smote.setPercentage(ratio);
			smote.setInputFormat(dataAutoencoder);
			smote.setRandomSeed(1);
			dataAutoencoder = Filter.useFilter(dataAutoencoder, smote);
		}

		if (type.equals(AutoencoderType.PARALLEL) || type.equals(AutoencoderType.BOTH)) {
			/* Autoencoder paralelo */
			roundData(dataAutoencoder);
			dataAutoencoder = getSelectedData(dataAutoencoder);
			dataAutoencoder = getParallelAutoencodedData(dataAutoencoder, dataAutoencoder.numAttributes());
			Integer pos = dataAutoencoder.numAttributes() - 1;
			dataAutoencoder.setClassIndex(pos);
			dataAutoencoder.setRelationName("Parallel Autoencoded");
		}

		if (type.equals(AutoencoderType.CASCADE) || type.equals(AutoencoderType.BOTH)) {
			/* Autoencoder piramidal */
			roundData(dataAutoencoder);
			dataAutoencoder = getSelectedMiddleData(dataAutoencoder);
			dataAutoencoder = getCascadeAutoencodedData(dataAutoencoder);
			dataAutoencoder.setRelationName("Cascade Autoencoded");
		}

		roundData(dataAutoencoder);
		dataAutoencoder.setClassIndex(dataAutoencoder.numAttributes() - 1);
		dataAutoencoder = getSelectedDataFin(dataAutoencoder);

		return dataAutoencoder;

	}

	private void roundData(Instances data) {
		for (int i = 0; i < data.numInstances(); i++) {
			for (int j = 0; j < data.numAttributes(); j++) {
				if (j != data.classIndex()) {
					double value = data.instance(i).value(j);
					data.instance(i).setValue(j, Math.round(value * Math.pow(10, 6)) / Math.pow(10.0, 6));
				}
			}
		}

	}

	private Instances getSelectedData(Instances aux) throws Exception {

		selectionIni = new AttributeSelection();

		selectionIni.setInputFormat(aux);

		return Filter.useFilter(aux, selectionIni);
	}

	private Instances getSelectedMiddleData(Instances aux) throws Exception {

		selectionMid = new AttributeSelection();
		selectionMid.setInputFormat(aux);

		return Filter.useFilter(aux, selectionMid);
	}

	private Instances getSelectedDataFin(Instances aux) throws Exception {

		selectionFin = new AttributeSelection();
		selectionFin.setInputFormat(aux);

		Instances result = Filter.useFilter(aux, selectionFin);
		return result;
	}

	private Instances getParallelAutoencodedData(Instances aux, Integer numFunctions) throws Exception {
		Instances result = null;

		for (int i = 1; i <= numFunctions; i++) {
			Instances auxiliar = new Instances(aux);
			Integer pos = auxiliar.classIndex();

			if (pos != -1) {
				auxiliar.setClassIndex(-1);
				auxiliar.deleteAttributeAt(pos);
			}

			MLPAutoencoder filtro = new MLPAutoencoder();
			filtrosParallel.add(filtro);
			filtro.setDoNotCheckCapabilities(true);
			filtro.setPoolSize(Runtime.getRuntime().availableProcessors() - 1);
			filtro.setNumThreads(filtro.getPoolSize());
			filtro.setNumFunctions(numFunctions);
			// filtro.setLambda(LAMBDA);
			filtro.setUseCGD(true);
			// filtro.setTolerance(TOLERANCE);
			filtro.setFilterType(new SelectedTag(AUTOENCODER_FILTER, MLPAutoencoder.TAGS_FILTER));
			filtro.setOutputInOriginalSpace(false);
			filtro.setInputFormat(auxiliar);
			auxiliar = Filter.useFilter(auxiliar, filtro);

			RenameAttribute renameFilter = new RenameAttribute();
			renameFilter.setFind("hidden");
			renameFilter.setReplace("ParallelLayer" + i + "_hidden");
			renameFilter.setInputFormat(auxiliar);
			auxiliar = Filter.useFilter(auxiliar, renameFilter);

			if (result != null) {
				result = Instances.mergeInstances(auxiliar, result);
			} else {
				result = auxiliar;
			}
		}

		result = Instances.mergeInstances(result, aux);

		return result;
	}

	private Instances getCascadeAutoencodedData(Instances aux) throws Exception {
		Instances result = null;

		Instances auxiliar = new Instances(aux);
		Integer pos = auxiliar.classIndex();
		if (pos != -1) {
			auxiliar.setClassIndex(-1);
			auxiliar.deleteAttributeAt(pos);
		}

		for (int i = aux.numAttributes() - 1; i >= 1; i--) {

			MLPAutoencoder filtro = new MLPAutoencoder();
			filtrosCascade.add(filtro);
			filtro.setDoNotCheckCapabilities(true);
			filtro.setPoolSize(Runtime.getRuntime().availableProcessors() - 1);
			filtro.setNumThreads(filtro.getPoolSize());
			// filtro.setLambda(LAMBDA);
			// filtro.setUseCGD(true);
			// filtro.setTolerance(TOLERANCE);
			filtro.setNumFunctions(i);
			filtro.setFilterType(new SelectedTag(AUTOENCODER_FILTER, MLPAutoencoder.TAGS_FILTER));
			filtro.setOutputInOriginalSpace(false);
			filtro.setInputFormat(auxiliar);
			auxiliar = Filter.useFilter(auxiliar, filtro);

			RenameAttribute renameFilter = new RenameAttribute();
			renameFilter.setFind("hidden");
			renameFilter.setReplace("CascadeLayer" + i + "_hidden");
			renameFilter.setInputFormat(auxiliar);
			auxiliar = Filter.useFilter(auxiliar, renameFilter);

			if (result != null) {
				result = Instances.mergeInstances(auxiliar, result);
			} else {
				result = auxiliar;
			}

		}

		result = Instances.mergeInstances(result, aux);

		return result;
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
		double[] autoencoded = cl.distributionForInstance(auxiliar.firstInstance());
		double[] origen = original.distributionForInstance(aux);
		return aux.classAttribute().isNominal() ? combine(origen, autoencoded) : average(origen, autoencoded);

	}

	private double[] average(double[] origen, double[] autoencoded) {
		double[] result = new double[origen.length];
		for (int i = 0; i < origen.length; i++) {
			result[i] = (origen[i] + autoencoded[i])/2;
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

	private Instances updateData(Instances test) throws Exception {
		test = Filter.useFilter(test, normalize);
		if (test.classAttribute().isNominal()) {
			test = Filter.useFilter(test, smote);
		}

		if (type.equals(AutoencoderType.PARALLEL) || type.equals(AutoencoderType.BOTH)) {
			/* Autoencoder paralelo */
			test = Filter.useFilter(test, selectionIni);
			test = applyAutoencodedData(test, false);
			test.setRelationName("Autoencoded");
		}

		if (type.equals(AutoencoderType.CASCADE) || type.equals(AutoencoderType.BOTH)) {
			/* Autoencoder piramidal */
			test = Filter.useFilter(test, selectionMid);
			test = applyAutoencodedData(test, true);
			test.setRelationName("Autoencoded");
		}

		return Filter.useFilter(test, selectionFin);
	}

	private Instances applyAutoencodedData(Instances test, boolean pure) throws Exception {
		Instances result = null;
		if (!pure) {
			Integer numFunctions = 1;
			/* Autoencoder paralelo */
			for (MLPAutoencoder auto : filtrosParallel) {
				Instances auxiliar = new Instances(test);
				Integer pos = auxiliar.classIndex();

				if (pos != -1) {
					auxiliar.setClassIndex(-1);
					auxiliar.deleteAttributeAt(pos);
				}

				auxiliar = Filter.useFilter(auxiliar, auto);

				RenameAttribute renameFilter = new RenameAttribute();
				renameFilter.setFind("hidden");
				renameFilter.setReplace("ParallelLayer" + numFunctions++ + "_hidden");
				renameFilter.setInputFormat(auxiliar);
				auxiliar = Filter.useFilter(auxiliar, renameFilter);

				if (result != null) {
					result = Instances.mergeInstances(auxiliar, result);
				} else {
					result = auxiliar;
				}

			}

			result = Instances.mergeInstances(result, test);
		} else {

			/* Autoencoder piramidal */
			Integer numFunctions = 1;

			Instances auxiliar = new Instances(test);
			Integer pos = auxiliar.classIndex();

			if (pos != -1) {
				auxiliar.setClassIndex(-1);
				auxiliar.deleteAttributeAt(pos);
			}
			for (MLPAutoencoder auto : filtrosCascade) {
				auxiliar = Filter.useFilter(auxiliar, auto);

				RenameAttribute renameFilter = new RenameAttribute();
				renameFilter.setFind("hidden");
				renameFilter.setReplace("CascadeLayer" + numFunctions++ + "_hidden");
				renameFilter.setInputFormat(auxiliar);
				auxiliar = Filter.useFilter(auxiliar, renameFilter);

				if (result != null) {
					result = Instances.mergeInstances(auxiliar, result);
				} else {
					result = auxiliar;
				}

			}

			result = Instances.mergeInstances(result, test);
		}

		return result;
	}

	public Capabilities getCapabilities() {
		return cl.getCapabilities();
	}

}
