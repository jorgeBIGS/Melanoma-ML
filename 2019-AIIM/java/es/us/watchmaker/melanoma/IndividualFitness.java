package es.us.watchmaker.melanoma;

import java.util.List;
import java.util.Random;
import java.util.SortedMap;
import java.util.function.Predicate;

import org.apache.commons.math3.util.Pair;
import org.uncommons.watchmaker.framework.FitnessEvaluator;

import minerva.utils.Utils;
import weka.core.Instances;

public class IndividualFitness implements FitnessEvaluator<IndividualAutoencoder> {

	private static final Predicate<Pair<Double, Double>> predicate;
	static {
		predicate = x->x.getFirst().isNaN();
	}
	private Instances training;
	private int[] atts;

	public IndividualFitness(Integer classIndex, Integer tIndex, Instances training) throws Exception {
		this.atts = new int[]{classIndex, tIndex};
		this.training = training;
		this.training.randomize(new Random());
		this.training.stratify(IndividualParameters.N_FOLDS);
	}

	public double getFitness(IndividualAutoencoder individual, List<? extends IndividualAutoencoder> l) {
		Double result = Double.NaN;
		Preprocessor preprocessor = new Preprocessor(atts);
		try {

			Instances data = preprocessor.preprocessData(training, individual);
			
			SortedMap<String,Pair<Double, Double>> map =  Utils.getCoxStatistics(data, atts[1], atts[0]);
			result = map.values().stream().filter(predicate).count() * 1.0;
			result += map.values().stream().filter(predicate.negate()).mapToDouble(x->x.getFirst()).sum();

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (result.isNaN() || result < 0) {
				result = isNatural() ? 0. : Double.MAX_VALUE;
			}
		}

		return result;
	}

	public boolean isNatural() {
		return false;
	}
	
	
	// for (int rep = 0; rep < IndividualParameters.N_REP; rep++) {
	// SimpleKMeans cl = new SimpleKMeans();
	// // cl.setNumClusters(individual.getK());
	// cl.buildClusterer(auxI);

	// int[][] matrix1 = new int[training.numClasses()][cl.getNumClusters()];
	// int[][] matrix2 = new int[cl.getNumClusters()][training.numClasses()];
	//
	// for (int i = 0; i < training.numInstances(); i++) {
	// Instance aux = preprocessor.preprocessInstance(training.instance(i),
	// individual);
	// Integer cluster = cl.clusterInstance(aux);
	// Integer classValue = (int) training.instance(i).classValue();
	// matrix1[classValue][cluster]++;
	// matrix2[cluster][classValue]++;
	// }
	//
	// double chi1 = Estadistiscas.calculaChiCuadrado(matrix1, matrix2,
	// training.numInstances());
	// double chi2 = Estadistiscas.calculaChiCuadrado(matrix2, matrix1,
	// training.numInstances());
	//
	// result += (Math.max(chi1, chi2) - Math.abs(chi1 - chi2));
}
