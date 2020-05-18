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

This model is pretty simple and straightforward and training is pretty quick. Basically running this is like running sklearn's NMF. Note that for this version the options for training method, the maximum GPU memory, and the activations for the factors and features (to ensure non-negativity) are meaningless. Training method is NADAM, it will take up the entire GPU, and it will use softplus activations. The important parameters are mu (supervision strength) and reg (L1 penalty on the supervision coefficients).
(https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)


### CSFA- Cross-spectral Mixture Kernel

This implements the CSFA model described in (https://papers.nips.cc/paper/5966-gp-kernels-for-cross-spectrum-analysis.pdf,http://papers.nips.cc/paper/7260-cross-spectral-factor-analysis.pdf) with the encoded supervision techniques. This model was implemented in Tensorflow 1.09 so it inherits a lot of the annoying features inherent in Tensorflow 1.09. The important values here are number of facotrs, eta, Q, and R. These are all described in the CSFA paper. Another important set of parameters are device and percGPU. These pick the GPU device to place the job on and the percentage of GPU memory respectively. Note that this is very memory intensive so even using the entire GPU will only permit a batch size < 1000 with C>8 or so.


