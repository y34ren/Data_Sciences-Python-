Regression Class Read Me-------------------

init - Regression() --

	create a empty model


Function:

	Fit_GRB(self,x,y,eta=1e-6 ,epochs=1e4, lambda1 = 0, lambda2 = 0,batch_sz = 1,show_curve = False)
		This function takes in a list of x and change it into PHI and performs regression with galsion radio bias
	
	Fit(self,x,y,eta=1e-6 ,epochs=1e4, lambda1 = 0, lambda2 = 0,batch_sz = 1,show_curve = False)
		
		This function taks a matrix of x and y and performs regular regression

	Fit2(self,x,y,eta=1e-6 ,epochs=1e4, lambda1 = 0, lambda2 = 0,batch_sz = 1,show_curve = False)
		This function performs a multivariate regression

	R2(self)
		Returns R-sqr of the test set

	predict(self, x)
		adds a 1 colum to the x list or matirx
		Returns y_hat of x input

	R2(y, y_hat):
		Return R-sqr given y and y_hat 		