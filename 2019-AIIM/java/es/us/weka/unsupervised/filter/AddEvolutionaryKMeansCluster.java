package es.us.weka.unsupervised.filter;

public class AddEvolutionaryKMeansCluster {

	// // private String atts;
	//
	// private List<Integer> metadata;
	//
	// public AddEvolutionaryKMeansCluster(String atts) {
	// this.atts = atts;
	// }
	//
	// public Instances determineOutputFormat(Instances in) throws Exception {
	// Instances result = new Instances(in);
	// Integer[] auxiliar = new Integer[IndividualParameters.MAX_K];
	// for (int i = 0; i < IndividualParameters.MAX_K; i++) {
	// auxiliar[i] = i;
	// }
	//
	// metadata = Arrays.stream(auxiliar).sorted().collect(Collectors.toList());
	// Attribute at = new Attribute("cluster", metadata.stream().map(x -> "" + (x +
	// 1)).collect(Collectors.toList()));
	// result.insertAttributeAt(at, result.numAttributes());
	//
	// return result;
	// }
	//
	// public String globalInfo() {
	// return "AddCluster evolutivo";
	// }
	//
	// public Instances process(Instances training) throws Exception {
	// // Instances result = determineOutputFormat(training);
	// //
	// CandidateFactory<IndividualAutoencoder> candidateFactory = new
	// IndividualFactoryImpl(training);
	//
	// List<EvolutionaryOperator<IndividualAutoencoder>> operators = new
	// LinkedList<EvolutionaryOperator<IndividualAutoencoder>>();
	// operators.add(new IndividualCrossover(IndividualParameters.N_CROSSP,
	// IndividualParameters.N_CHILDREN));
	// operators.add(new IndividualMutation());
	//
	// EvolutionaryOperator<IndividualAutoencoder> pipeline = new
	// EvolutionPipeline<IndividualAutoencoder>(operators);
	//
	// FitnessEvaluator<IndividualAutoencoder> fitnessEvaluator = new
	// IndividualFitness(0, 1, training);
	// SelectionStrategy<Object> selectionStrategy = new RouletteWheelSelection();
	//
	// EvolutionEngine<IndividualAutoencoder> engine = new
	// GenerationalEvolutionEngine<IndividualAutoencoder>(
	// candidateFactory, pipeline, fitnessEvaluator, selectionStrategy,
	// new Random((long) (Math.random() * Long.MAX_VALUE)));
	//
	// engine.addEvolutionObserver(new
	// EvolutionGenericObserver<IndividualAutoencoder>());
	//
	// IndividualAutoencoder individual = engine.evolve(IndividualParameters.N_POP,
	// IndividualParameters.N_ELITISM,
	// new GenerationCount(IndividualParameters.N_GEN), new TargetFitness(0.,
	// false));
	//
	// Preprocessor preprocessor = new Preprocessor(new int[] { 0, 1 });
	//
	// Instances trainingAux = preprocessor.preprocessData(training, individual);
	//
	// return trainingAux;
	//
	// }
}

// SimpleKMeans kmeans = new SimpleKMeans();
// kmeans.setNumClusters(individual.getK());
// kmeans.buildClusterer(trainingAux);
//
// for (int i = 0; i < trainingAux.numInstances(); i++) {
// Instance aux = trainingAux.instance(i);
// Integer cluster = kmeans.clusterInstance(aux);
//
// result.instance(i).setValue(result.numAttributes() - 1,
// metadata.indexOf(cluster));
//
// }
