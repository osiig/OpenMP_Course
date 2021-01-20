import numpy as np

class GaussianProcess:
    def __init__(self,x,y,sigma=None, l=None, k0=None):

        # initialise data
        self.x = x
        self.y = y

        # initialise hyper-parameters
        if sigma == None: 
            self.sigma = sigma
        else:
            self.sigma = sigma 
        
        self.l     = l     # width of gaussian
        self.k0    = k0    # magnitude of gaussian

        # initialise other stuff
        self.N = len(x)

        # Initialise properties
        self.C    = None
        self.Cinv = None
        self.yp   = None

    def kernel(self,x,xm): #function for the gaussian of mean xm and width l evaluated in x
        return np.exp(-(x-xm)**2/(2*self.l**2))
        
    def prior(self,x):
        return np.zeros(x.shape)

    def get_prior(self):
        if self.yp == None:
            self.yp = self.prior(self.x)

    def get_C(self): # Get C-matrix (C=K + sigma^2 * I_n)
        if self.C == None:
            self.X = np.meshgrid(self.x,self.x)[0]     
            self.K = self.kernel(self.X,np.transpose(self.X))
            self.C = self.K + self.sigma**2 * np.identity(self.N)        

    def get_Cinv(self): # Get inverted C-matrix
        if self.Cinv == None:
            self.Cinv = np.linalg.inv(self.C)

    def prediction(self,xf):
        # takes input data (x,y) and outputs fitted values (xf,yf)
        
        self.get_C()
        self.get_Cinv()
        self.get_prior()

        Xf, X= np.meshgrid(xf,self.x)
        k    = self.kernel(Xf,X)
        
        # Prediction
        yf = self.prior(xf) + np.linalg.multi_dot([k.T,self.Cinv,(self.y-self.yp)])
        # Variance
        vf = self.k0 - np.einsum('ij,ji->i',k.T,np.matmul(self.Cinv,k))
        
        return yf , vf