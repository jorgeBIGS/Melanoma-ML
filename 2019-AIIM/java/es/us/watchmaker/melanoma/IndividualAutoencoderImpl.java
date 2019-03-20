package es.us.watchmaker.melanoma;

import java.util.Random;

public class IndividualAutoencoderImpl implements IndividualAutoencoder {


	private Integer hiddens, filter;

	private Double lambda, tolerance;

	public IndividualAutoencoderImpl(Integer numGenes, Random r) {
		hiddens = r.nextInt(numGenes) + 1;
		filter = r.nextInt(IndividualParameters.MAX_AUTOENCODER_TYPE);
		lambda = r.nextDouble() * IndividualParameters.MAX_AUTOENCODER_LAMBDA;
		tolerance = r.nextDouble() * IndividualParameters.MAX_AUTOENCODER_TOLERANCE;
	}
	
	public IndividualAutoencoderImpl(IndividualAutoencoder i1) {
		hiddens = i1.getAutoencoderNumHidden();
		filter = i1.getAutoencoderFilter();
		lambda = i1.getAutoencoderLambda();
		tolerance = i1.getAutoencoderTolerance();
	}

	public IndividualAutoencoderImpl(IndividualAutoencoder i1, IndividualAutoencoder i2, Random random) {
		this(i1);
		hiddens = random.nextBoolean() ? hiddens : i2.getAutoencoderNumHidden();
		filter = random.nextBoolean() ? filter : i2.getAutoencoderFilter();
		lambda = random.nextBoolean() ? lambda : i2.getAutoencoderLambda();
		tolerance = random.nextBoolean() ? tolerance : i2.getAutoencoderTolerance();
	}

	
	public void mutate(Random r) {

		if (r.nextDouble() < IndividualParameters.PROB_MUTATION) {
			hiddens = r.nextInt(hiddens) + 1;
		}
		if (r.nextDouble() < IndividualParameters.PROB_MUTATION) {
			filter = r.nextInt(IndividualParameters.MAX_AUTOENCODER_TYPE);
		}
		if (r.nextDouble() < IndividualParameters.PROB_MUTATION) {
			lambda = (1 - r.nextDouble()) * IndividualParameters.MAX_AUTOENCODER_LAMBDA;
		}
		if (r.nextDouble() < IndividualParameters.PROB_MUTATION) {
			tolerance = (1 - r.nextDouble()) * IndividualParameters.MAX_AUTOENCODER_TOLERANCE;
		}
	}

	public Integer getAutoencoderFilter() {
		return filter;
	}

	public Integer getAutoencoderNumHidden() {
		return hiddens;
	}

	public Double getAutoencoderLambda() {
		return lambda;
	}

	public Double getAutoencoderTolerance() {
		return tolerance;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((filter == null) ? 0 : filter.hashCode());
		result = prime * result + ((hiddens == null) ? 0 : hiddens.hashCode());
		result = prime * result + ((lambda == null) ? 0 : lambda.hashCode());
		result = prime * result + ((tolerance == null) ? 0 : tolerance.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		IndividualAutoencoderImpl other = (IndividualAutoencoderImpl) obj;
		if (filter == null) {
			if (other.filter != null)
				return false;
		} else if (!filter.equals(other.filter))
			return false;
		if (hiddens == null) {
			if (other.hiddens != null)
				return false;
		} else if (!hiddens.equals(other.hiddens))
			return false;
		if (lambda == null) {
			if (other.lambda != null)
				return false;
		} else if (!lambda.equals(other.lambda))
			return false;
		if (tolerance == null) {
			if (other.tolerance != null)
				return false;
		} else if (!tolerance.equals(other.tolerance))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "IndividualAutoencoderImpl [hiddens=" + hiddens + ", filter=" + filter + ", lambda=" + lambda
				+ ", tolerance=" + tolerance + "]";
	}
	
	
	// private List<Double> genes;

	// private Integer k;
	
	// genes = Lists.newArrayList();
	// inicializaGenes(numGenes);
	// genes.set(r.nextInt(numGenes), 1.0);
	// k = r.nextInt(IndividualParameters.MAX_K) + 2;

	// private void inicializaGenes(Integer numGenes) {
	// for (int i = 0; i < numGenes; i++) {
	// genes.add(0.);
	// }
	// }

	// genes = Lists.newArrayList(i1.getFeatures());
	// k = i1.getK();

	// for (int i = 0; i < i1.getFeatures().size(); i++) {
	// if (random.nextBoolean()) {
	// genes.set(i, i2.getFeatures().get(i));
	// }
	// }
	// k = random.nextBoolean() ? i1.getK() : i2.getK();

	// public List<Double> getGenes() {
	// return genes;
	// }


	// if (r.nextBoolean()) {
	// for (int k = 0; k < getFeatures().size(); k++) {
	// if (r.nextDouble() < IndividualParameters.PROB_MUTATION) {
	// getGenes().set(k, r.nextDouble());
	// }
	// }
	// } else {

	// if (r.nextDouble() < IndividualParameters.PROB_MUTATION) {
	// k = r.nextInt(IndividualParameters.MAX_K) + 2;
	// }

	// }

	// public List<Double> getFeatures() {
	// return new ArrayList<>(genes);
	// }
	//
	// public Integer getK() {
	// return k;
	// }


}
