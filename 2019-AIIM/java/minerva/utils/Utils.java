package minerva.utils;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.commons.math3.util.Pair;

import javastat.survival.regression.CoxRegression;
import weka.core.Instances;

public class Utils {
	
	private static double[] toArray(List<Double> genes) {
		double[] result = new double[genes.size()];
		for (int i = 0; i < genes.size(); i++) {
			result[i] = genes.get(i);
		}
		return result;
	}

	private static List<Double> toList(double[] distro) {
		List<Double> result = new LinkedList<>();
		for (double d : distro) {
			result.add(d);
		}
		return result;
	}


	public static SortedMap<String, Pair<Double, Double>> getCoxStatistics(Instances done, Integer tAtt,
			Integer classIndex) {

		double[] taim = done.attributeToDoubleArray(tAtt);
		double[] aim = done.attributeToDoubleArray(classIndex);

		Map<String, List<Double>> attributes = new HashMap<>();
		SortedMap<String, Pair<Double, Double>> result = new TreeMap<>();

		for (int i = 0; i < done.numAttributes(); i++) {
			if (i != tAtt && i != classIndex) {
				attributes.put(done.attribute(i).name(), toList(done.attributeToDoubleArray(i)));
			}
		}

		for (String key : attributes.keySet()) {
			try {
				CoxRegression testclass1 = new CoxRegression(taim, aim, toArray(attributes.get(key)));
				Pair<Double, Double> value = new Pair<>(testclass1.pValue[0],
						Math.pow(Math.E, testclass1.coefficients[0]));
				result.put(key, value);
			} catch (RuntimeException e) {
				Pair<Double, Double> value = new Pair<>(Double.NaN, Double.NaN);
				result.put(key, value);
			}
		}

		return result;

	}
}
