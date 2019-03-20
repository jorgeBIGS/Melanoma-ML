package es.us.watchmaker.melanoma;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.MLPAutoencoder;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class Preprocessor {
	private Remove filter;
	private ReplaceMissingValues filter0;
	private MLPAutoencoder filter1;
	private Discretize filter2;
	private NominalToBinary filter3;
	private int[] removed;

	public Preprocessor(int[] atts) {
		this.removed = atts;
	}

	public Instances preprocessData(Instances todos) throws Exception {
		Instances newInstances = new Instances(todos);

		newInstances.setClassIndex(-1);
		filter = new Remove();
		filter.setAttributeIndicesArray(removed);
		filter.setInvertSelection(false);
		filter.setInputFormat(newInstances);

		newInstances = Filter.useFilter(newInstances, filter);

		return newInstances;
	}

	public Instances preprocessData(Instances todos, IndividualAutoencoder best) throws Exception {
		Instances newInstances = new Instances(todos);

		newInstances = preprocessData(newInstances);

		filter0 = new ReplaceMissingValues();
		filter0.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter0);

		filter1 = new MLPAutoencoder();
		// filter1.setFilterType(new SelectedTag(best.getAutoencoderFilter(),
		// MLPAutoencoder.TAGS_FILTER));
		 filter1.setFilterType(new SelectedTag(MLPAutoencoder.FILTER_NORMALIZE,
		 MLPAutoencoder.TAGS_FILTER));
		// filter1.setNumFunctions(best.getAutoencoderNumHidden());
		// filter1.setLambda(best.getAutoencoderLambda());
		// filter1.setTolerance(best.getAutoencoderTolerance());
		// filter1.setUseContractiveAutoencoder(true);
		filter1.setOutputInOriginalSpace(true);
		filter1.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter1);

		newInstances.insertAttributeAt(todos.attribute(removed[0]), removed[0]);
		newInstances.insertAttributeAt(todos.attribute(removed[1]), removed[1]);

		for (int i = 0; i < newInstances.numInstances(); i++) {
			Instance ins = newInstances.instance(i);
			ins.setValue(removed[0], todos.instance(i).value(removed[0]));
			ins.setValue(removed[1], todos.instance(i).value(removed[1]));
		}

		newInstances.setClassIndex(removed[0]);

		filter2 = new Discretize();
		filter2.setAttributeIndicesArray(removed);
		filter2.setInvertSelection(true);
		// filter2.setFindNumBins(true);
		filter2.setMakeBinary(true);
		filter2.setUseEqualFrequency(true);
		filter2.setBins(2);
		// filter2.setUseBinNumbers(true);
		filter2.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter2);
		newInstances.setClassIndex(removed[0]);

		filter3 = new NominalToBinary();
		filter3.setInputFormat(newInstances);
		newInstances = Filter.useFilter(newInstances, filter3);

		return newInstances;
	}

	// public Instance preprocessInstance(Instance una, IndividualAutoencoder best)
	// throws Exception {
	// Instances newInstances = new Instances(una.dataset(), 0);
	// newInstances.add(una);
	//
	// newInstances.setClassIndex(-1);
	// newInstances = Filter.useFilter(newInstances, filter0);
	//
	// newInstances = Filter.useFilter(newInstances, filter1);
	//
	// for (int i = 0; i < newInstances.numInstances(); i++) {
	// Instance ins = newInstances.instance(i);
	// for (int j = 0; j < ins.numAttributes() - 1; j++) {
	// ins.setValue(j, ins.value(j) * best.getFeatures().get(j));
	// }
	// }
	//
	// return newInstances.firstInstance();
	// }
}
