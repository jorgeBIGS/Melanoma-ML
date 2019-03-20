package es.us.indices;

import java.util.Arrays;
import java.util.Random;

public class ConfusionEntropy {

	public static void main(String[] args) {
		Random r = new Random();
		Integer dim1;
		for (int k = 0; k < 1000; k++) {
			dim1 = r.nextInt(10)+2;
			int [][] array1 = new int[2][dim1];
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < array1[i].length; j++) {
					array1[i][j] = r.nextInt(20);
				}
			}
			System.out.println(getCEN(array1));
		}
		
	}

	public static Double getCEN(int[][] array) {
		Double result = 0.0;

		Integer sumaTotal = getSumaTotal(array);
		Double[] p = getProbabilitiesByLabel(array, sumaTotal);
		Double[] cen = getCENByLabel(array, sumaTotal);
		for (int i = 0; i < p.length; i++) {
			result += p[i] * cen[i];
		}
		return result;
	}

	private static Double[] getCENByLabel(int[][] array, Integer sumaTotal) {
		Double[] result = new Double[array.length];
		Arrays.fill(result, 0.0);
		double[][] probs = getProbabilities(array, sumaTotal);
		Double probTrans = Math.log(2 * (array.length - 1));
		for (int j = 0; j < array.length; j++) {
			for (int k = 0; k < array[j].length; k++) {
				if (k != j) {
					result[j] += (probs[j][k] == 0 ? 0 : probs[j][k] * Math.log(probs[j][k]) / probTrans);
				}
			}
			for (int k = 0; k < array.length; k++) {
				if (k != j) {
					try {
						result[j] += (probs[k][j] == 0 ? 0 : probs[k][j] * Math.log(probs[k][j]) / probTrans);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}

			result[j] = -result[j];
		}

		return result;
	}

	private static double[][] getProbabilities(int[][] array, Integer sumaTotal) {
		double[][] result = new double[array.length][array[0].length];
		for (int j = 0; j < array.length; j++) {
			for (int k = 0; k < array[j].length; k++) {
				result[j][k] = 1.0 * array[j][k] / sumaTotal;
			}
		}
		return result;
	}

	private static Double[] getProbabilitiesByLabel(int[][] array, Integer sumaTotal) {
		Double[] result = new Double[array.length];
		Arrays.fill(result, 0.0);

		for (int i = 0; i < array.length; i++) {
			Double subResult = 0.0;
			for (int j = 0; j < array[i].length; j++) {
				subResult += array[i][j];
			}

			for (int k = 0; k < array.length; k++) {
				try {
					subResult += array[k][i];
				} catch (Exception e) {
					System.out.println(e);
				}
			}

			result[i] = subResult.doubleValue() / sumaTotal;
		}

		return result;
	}

	private static Integer getSumaTotal(int[][] array) {
		Integer result = 0;

		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[i].length; j++) {
				result += array[i][j];
			}
		}

		return result;
	}

}
