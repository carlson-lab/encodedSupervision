# encodedSupervision

This contains the code associated with the paper by Talbot, Dunson, Dzirasa and Carlson. The manuscript is now available on Arxiv (https://arxiv.org/pdf/2004.05209.pdf).

## Enocded Supervision - Finding Brain Networks

Previous research has demonstrated that traditional supervised factor models are susceptible to model misspecification, particularly in the number of factors. Often the variance of the 

### Code requirements

All code is implemented in Python 3. Unfortunately the current implementation of the non-negative matrix factorization requires Tensorflow 2.0 wheras the implementation of the Cross-spectral mixture kernel requires Tensorflow 1.09. Otherwise the requirements are the standard libraries that come with Anaconda. To the best of my ability I have made the UI for the models align with the usage for factor models in Sklearn (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition). Unfortunately I haven't had the time to actually have the modelse extend many of their libraries and the code doesn't check input formats, however the usage is very similar as 

model = ModelClass(number_of_components,keyWordArg1=value1,keyWordArg2=value2)
Scores_training = model.fit_transform(X_train,Y_train)
Scores_test = model.transform(X_test)

loadings = model.components_

#Encoder = softmax(AX+B)
A = model.A_enc
B = model.B_enc

Model-specific considerations will be described below.

### CSFA- Non-negative Matrix Factorization (NMF)

(https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)


### CSFA- Cross-spectral Mixture Kernel

This implements the CSFA model described in (https://papers.nips.cc/paper/5966-gp-kernels-for-cross-spectrum-analysis.pdf,http://papers.nips.cc/paper/7260-cross-spectral-factor-analysis.pdf) with the encoded supervision techniques. This model was implemented in Tensorflow 1.09 so it inherits some of the 


