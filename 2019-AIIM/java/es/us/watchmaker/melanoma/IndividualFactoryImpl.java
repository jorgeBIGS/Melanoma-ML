package es.us.watchmaker.melanoma;

import java.util.Random;

import org.uncommons.watchmaker.framework.factories.AbstractCandidateFactory;

import weka.core.Instances;

public class IndividualFactoryImpl extends
		AbstractCandidateFactory<IndividualAutoencoder> {

	private Integer numGenes;

	public IndividualFactoryImpl(Instances training) {
		numGenes = training.numAttributes() - 1;
	}

	public IndividualAutoencoder generateRandomCandidate(Random r) {
		return new IndividualAutoencoderImpl(numGenes, r);
	}

}
