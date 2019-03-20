package es.us.watchmaker.melanoma;

import java.util.List;
import java.util.Random;

import org.uncommons.watchmaker.framework.EvolutionaryOperator;

import com.google.common.collect.Lists;

public class IndividualMutation implements EvolutionaryOperator<IndividualAutoencoder> {

	public List<IndividualAutoencoder> apply(List<IndividualAutoencoder> l, Random r) {
		List<IndividualAutoencoder> result = Lists.newLinkedList();
		for (IndividualAutoencoder ind : l) {
			IndividualAutoencoder gen = new IndividualAutoencoderImpl(ind);

			gen.mutate(r);

			result.add(gen);
		}
		return result;
	}

}
