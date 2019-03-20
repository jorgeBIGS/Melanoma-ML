package es.us.indices;

import weka.core.Instance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Cluster implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private List<Instance> puntos = new ArrayList<Instance>();
	private Instance centroide;
	private boolean termino = false;

	public Instance getCentroide() {
		return centroide;
	}

	public void setCentroide(Instance centroide) {
		this.centroide = centroide;
	}

	public List<Instance> getInstances() {
		return puntos;
	}

	public List<Instance> getPuntos() {
		return puntos;
	}

	public boolean isTermino() {
		return termino;
	}

	public void setTermino(boolean termino) {
		this.termino = termino;
	}

	public void limpiarInstances() {
		puntos.clear();
	}

	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((centroide == null) ? 0 : centroide.hashCode());
		return result;
	}

	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Cluster other = (Cluster) obj;
		if (centroide == null) {
			if (other.centroide != null)
				return false;
		} else if (!centroide.equals(other.centroide))
			return false;
		return true;
	}

	public String toString() {
		return centroide.toString();
	}
}
