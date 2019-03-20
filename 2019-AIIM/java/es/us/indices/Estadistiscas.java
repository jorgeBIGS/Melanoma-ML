package es.us.indices;

import weka.core.Utils;

public class Estadistiscas {

	public static Double calculaChiCuadrado(int[][] matrixBase, int[][] matrixApoyo, Integer total) {
		Double result = 0.0;

		for (int i = 0; i < matrixBase.length; i++) {
			Double sumaFila = Utils.sum(matrixBase[i])*1.0/total;
			for (int j = 0; j < matrixApoyo.length; j++) {
				Double sumaColumna = Utils.sum(matrixApoyo[j])*1.0/total;
				Double esperado = (1.0 * sumaFila * sumaColumna);
				if (esperado > 0)
					result += Math.pow(matrixBase[i][j]/(sumaColumna*total) - esperado, 2.0) / esperado;
			}
		}

		return result;
	}

	public static void main(String[] args) {
		int[][] array1 = { { 5, 11, 7 }, { 20, 32, 3 } };
		int[][] array2 = { { 5, 20 }, { 11, 32 }, { 7, 3 } };
		
		

		System.out.println(calculaChiCuadrado(array1, array2, 78));
		System.out.println(calculaChiCuadrado(array2, array1, 78));

	}

}
