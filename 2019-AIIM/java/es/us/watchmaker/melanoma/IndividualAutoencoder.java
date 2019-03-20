package es.us.watchmaker.melanoma;

import java.util.Random;

public interface IndividualAutoencoder{
	// List<Double> getFeatures();
	//
	// Integer getK();
	
	Integer getAutoencoderFilter();
	
	Integer getAutoencoderNumHidden();
	
	Double getAutoencoderLambda();
	
	Double getAutoencoderTolerance();

	void mutate(Random r);
}
