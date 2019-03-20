package es.us.watchmaker.general;

import java.util.logging.Logger;

import org.uncommons.watchmaker.framework.EvolutionObserver;
import org.uncommons.watchmaker.framework.PopulationData;

public class EvolutionGenericObserver<T> implements EvolutionObserver<T> {

	private static Logger logger;

	public EvolutionGenericObserver() {
		logger = Logger.getLogger(EvolutionGenericObserver.class.getName());
	}

	public void populationUpdate(PopulationData<? extends T> p) {
		logger.info("Generación:  " + p.getGenerationNumber() + " -> " + p.getBestCandidate() + " = "
				+ p.getBestCandidateFitness() );
	}

}
