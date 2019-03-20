package minerva.melanoma.test;

import java.io.File;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.uncommons.watchmaker.framework.CandidateFactory;
import org.uncommons.watchmaker.framework.EvolutionEngine;
import org.uncommons.watchmaker.framework.EvolutionaryOperator;
import org.uncommons.watchmaker.framework.FitnessEvaluator;
import org.uncommons.watchmaker.framework.GenerationalEvolutionEngine;
import org.uncommons.watchmaker.framework.SelectionStrategy;
import org.uncommons.watchmaker.framework.operators.EvolutionPipeline;
import org.uncommons.watchmaker.framework.selection.RouletteWheelSelection;
import org.uncommons.watchmaker.framework.termination.GenerationCount;
import org.uncommons.watchmaker.framework.termination.TargetFitness;

import es.us.watchmaker.general.EvolutionGenericObserver;
import es.us.watchmaker.melanoma.IndividualAutoencoder;
import es.us.watchmaker.melanoma.IndividualCrossover;
import es.us.watchmaker.melanoma.IndividualFactoryImpl;
import es.us.watchmaker.melanoma.IndividualFitness;
import es.us.watchmaker.melanoma.IndividualMutation;
import es.us.watchmaker.melanoma.IndividualParameters;
import es.us.watchmaker.melanoma.Preprocessor;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class TestEvolutionaryCox {
	private static final String DATA_PATH = "data/macarena.arff";

	public static void main(String[] args) throws Exception {
		ArffLoader csv = new ArffLoader();
		csv.setFile(new File(DATA_PATH));
		Instances train = csv.getDataSet();

		// train.deleteAttributeAt(1);
		// train.deleteAttributeAt(1);
		train.setClassIndex(0);
		
		System.out.println(train);

		Instances done = preprocess(train);

		System.out.println(done);

	}
	
	private static Instances preprocess(Instances training) throws Exception {
		CandidateFactory<IndividualAutoencoder> candidateFactory = new IndividualFactoryImpl(training);

		List<EvolutionaryOperator<IndividualAutoencoder>> operators = new LinkedList<EvolutionaryOperator<IndividualAutoencoder>>();
		operators.add(new IndividualCrossover(IndividualParameters.N_CROSSP, IndividualParameters.N_CHILDREN));
		operators.add(new IndividualMutation());

		EvolutionaryOperator<IndividualAutoencoder> pipeline = new EvolutionPipeline<IndividualAutoencoder>(operators);

		FitnessEvaluator<IndividualAutoencoder> fitnessEvaluator = new IndividualFitness(0, 1, training);
		SelectionStrategy<Object> selectionStrategy = new RouletteWheelSelection();

		EvolutionEngine<IndividualAutoencoder> engine = new GenerationalEvolutionEngine<IndividualAutoencoder>(
				candidateFactory, pipeline, fitnessEvaluator, selectionStrategy,
				new Random((long) (Math.random() * Long.MAX_VALUE)));

		engine.addEvolutionObserver(new EvolutionGenericObserver<IndividualAutoencoder>());

		IndividualAutoencoder individual = engine.evolve(IndividualParameters.N_POP, IndividualParameters.N_ELITISM,
				new GenerationCount(IndividualParameters.N_GEN),
				new TargetFitness(fitnessEvaluator.isNatural() ? 0. : Double.MAX_VALUE, fitnessEvaluator.isNatural()));

		Preprocessor preprocessor = new Preprocessor(new int[] { 0, 1 });

		return preprocessor.preprocessData(training, individual);
	}

}
