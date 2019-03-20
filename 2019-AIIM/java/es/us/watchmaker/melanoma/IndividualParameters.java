package es.us.watchmaker.melanoma;

import weka.filters.unsupervised.attribute.MLPAutoencoder;

public class IndividualParameters {
	// TODO: Ajustar esto
	public static final Integer N_REP = 5;
	public static final Integer N_FOLDS = 5;
	public static final Integer N_POP = 50;
	public static final Integer N_GEN = 1;
	public static final Integer N_CHILDREN = 2;
	public static final Integer N_CROSSP = 1;
	public static final Integer N_ELITISM = 2;
	public static final Double PROB_MUTATION = 0.2;

	public static final Integer MAX_K = 20;

	public static final Integer MAX_AUTOENCODER_TYPE = MLPAutoencoder.TAGS_FILTER.length;
	public static final Double MAX_AUTOENCODER_LAMBDA = 0.1;
	public static final Double MAX_AUTOENCODER_TOLERANCE = Math.pow(10, -5);
}
