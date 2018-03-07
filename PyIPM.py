import numpy as np

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
         
        from pyDOE import fullfact
        self.exponentList=fullfact(np.int32((self.polynomialDegree+1)*np.ones((1,self.NFeatures))[0]))
        self.Nexponents=self.exponentList.shape[0]       
        
        self.InputScale=np.mean(np.abs(trainingInput),axis=0);
        trainingInput=trainingInput/self.InputScale;   
        
        basis=np.zeros((self.NDataPoints,self.Nexponents))
        
        for j,exponentVector in enumerate(self.exponentList):
            for i,inputSample in enumerate(trainingInput):
                basis[i,j]=np.prod(inputSample**exponentVector)
        
        basisSum=np.mean(np.absolute(basis), axis=0)
        objective=np.concatenate((-basisSum,basisSum))  
        
        constraintMat=np.zeros((2*self.NDataPoints+self.Nexponents,2*self.Nexponents))
        
        constraintMat[:self.NDataPoints,:self.Nexponents]=-(basis-np.absolute(basis))/2
        constraintMat[self.NDataPoints:-self.Nexponents,:self.Nexponents]=(basis+np.absolute(basis))/2
        constraintMat[:self.NDataPoints,self.Nexponents:]=-(basis+np.absolute(basis))/2
        constraintMat[self.NDataPoints:-self.Nexponents,self.Nexponents:]=(basis-np.absolute(basis))/2
        
        constraintMat[-self.Nexponents:,:self.Nexponents]=np.eye(self.Nexponents)
        constraintMat[-self.Nexponents:,self.Nexponents:]=-np.eye(self.Nexponents)
            
        b=np.zeros(((2*self.NDataPoints+self.Nexponents),1))
        b[:self.NDataPoints,0]=-trainingOutput
        b[self.NDataPoints:2*self.NDataPoints,0]=trainingOutput
        
        from cvxopt import matrix, solvers
              
        sol=solvers.lp(matrix(objective),matrix(constraintMat),matrix(b))
        
        self.paramVec=np.array(sol['x'])
        
        return self        
    def predict(self,testInputs):
        try:
            getattr(self, "paramVec")
        except AttributeError:
            raise RuntimeError("You must train IPM before predicting data!")
            
        assert(testInputs.shape[1]==self.NFeatures),'The provided test data has the wrong number of features'
        
        NTestPoints=testInputs.shape[0]
        
        testInputs=testInputs/self.InputScale
        
        basis=np.zeros((self.NDataPoints,self.Nexponents))
        
        for j,exponentVector in enumerate(self.exponentList):
            for i,inputSample in enumerate(testInputs):
                basis[i,j]=np.prod(inputSample**exponentVector)      
        
        constraintMat=np.zeros((2*self.NDataPoints,2*self.Nexponents))
        
        constraintMat[:self.NDataPoints,:self.Nexponents]=-(basis-np.absolute(basis))/2
        constraintMat[self.NDataPoints:,:self.Nexponents]=(basis+np.absolute(basis))/2
        constraintMat[:self.NDataPoints,self.Nexponents:]=-(basis+np.absolute(basis))/2
        constraintMat[self.NDataPoints:,self.Nexponents:]=(basis-np.absolute(basis))/2        
        
        out=np.dot(constraintMat,self.paramVec)
        
        return(-out[:NTestPoints],out[NTestPoints:])
    def getModelReliability(self,confidence=1-10**-6):            
        if confidence<0 or confidence>1:
            print('Invalid confidence parameter value')
        else:
            return(1-2*self.Nexponents/((self.NDataPoints+1)*confidence))
                