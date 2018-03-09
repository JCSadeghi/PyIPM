import numpy as np
from sklearn.preprocessing import PolynomialFeatures
#==============================================================================
#     This class is a port of the MATLAB code IntervalPredictorModel from
#     OpenCossan
#
#     2018, Jonathan Sadeghi, COSSAN Working Group,
#     University~of~Liverpool, United Kingdom
#     See also:  http://cossan.co.uk/wiki/index.php/@IntervalPredictorModel
#==============================================================================

class PyIPM:
    def __init__(self, polynomialDegree=1):
        self.polynomialDegree = polynomialDegree
        assert (type(self.polynomialDegree) == int), 'polynomialDegree parameter must be integer'
    def fit(self,trainingInput,trainingOutput):
        self.NFeatures=trainingInput.shape[1]
        self.NDataPoints=trainingInput.shape[0]

        assert(trainingOutput.shape==(self.NDataPoints,)),'Number of input examples must equal number of output examples'

        self.InputScale=np.mean(np.abs(trainingInput),axis=0);
        trainingInput=trainingInput/self.InputScale;

        poly = PolynomialFeatures(self.polynomialDegree)
        basis=poly.fit_transform(trainingInput)
        self.Nterms=basis.shape[1]

        basisSum=np.mean(np.absolute(basis), axis=0)
        objective=np.concatenate((-basisSum,basisSum))

        constraintMatrix=np.zeros((2*self.NDataPoints+self.Nterms,2*self.Nterms))

        constraintMatrix[:self.NDataPoints,:self.Nterms]=-(basis-np.absolute(basis))/2
        constraintMatrix[self.NDataPoints:-self.Nterms,:self.Nterms]=(basis+np.absolute(basis))/2
        constraintMatrix[:self.NDataPoints,self.Nterms:]=-(basis+np.absolute(basis))/2
        constraintMatrix[self.NDataPoints:-self.Nterms,self.Nterms:]=(basis-np.absolute(basis))/2

        constraintMatrix[-self.Nterms:,:self.Nterms]=np.eye(self.Nterms)
        constraintMatrix[-self.Nterms:,self.Nterms:]=-np.eye(self.Nterms)

        b=np.zeros(((2*self.NDataPoints+self.Nterms),1))
        b[:2*self.NDataPoints,0]=np.hstack((-trainingOutput,trainingOutput))

        from cvxopt import matrix, solvers

        sol=solvers.lp(matrix(objective),matrix(constraintMatrix),matrix(b))

        self.paramVec=np.array(sol['x'])

        return self
    def predict(self,testInput):
        try:
            getattr(self, "paramVec")
        except AttributeError:
            raise RuntimeError("You must train IPM before predicting data!")

        assert(testInput.shape[1]==self.NFeatures),'The provided test data has the wrong number of features'

        NTestPoints=testInput.shape[0]

        testInput=testInput/self.InputScale

        poly = PolynomialFeatures(self.polynomialDegree)
        basis=poly.fit_transform(testInput)

        upperBound=0.5*np.dot(np.hstack((basis-np.absolute(basis),basis+np.absolute(basis))),self.paramVec)
        lowerBound=0.5*np.dot(np.hstack((basis+np.absolute(basis),basis-np.absolute(basis))),self.paramVec)

        return(upperBound,lowerBound)
    def getModelReliability(self,confidence=1-10**-6):
        if confidence<0 or confidence>1:
            print('Invalid confidence parameter value')
        else:
            return(1-2*self.Nterms/((self.NDataPoints+1)*confidence))
