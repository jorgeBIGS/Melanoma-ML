package es.us.watchmaker.melanoma;

import java.util.List;
import java.util.Random;

import org.uncommons.watchmaker.framework.operators.AbstractCrossover;

import com.google.common.collect.Lists;

public class IndividualCrossover extends
		AbstractCrossover<IndividualAutoencoder> {

	private Integer numChildren;

	public IndividualCrossover(int crossoverPoints, Integer nChildren) {
		super(crossoverPoints);
		numChildren = nChildren;
	}

	protected List<IndividualAutoencoder> mate(IndividualAutoencoder i1,
			IndividualAutoencoder i2, int arg2, Random r) {
		List<IndividualAutoencoder> result = Lists.newArrayList();
		for (int i = 0; i < numChildren; i++) {
			// Cruce uniforme
			result.add(new IndividualAutoencoderImpl(i1, i2, r));
		}
		return result;

	}

}
