Classification Class Read Me-------------------

init - Classification() --

	create a empty model


Function:

	
	Fit(self,x,y,eta=1e-6 ,epochs=1e4, lambda1 = 0, lambda2 = 0,batch_sz = 1,show_curve = False)
		
		This function taks a matrix of x and y and performs logistic regression (y does not need to be one_hot_encode before hand)

	accuracy(self)
		Returns the accuracy of the test set

	predict(self, x)
		adds a 1 colum to the x list or matirx
		Returns p_hat of x input

	accuracy(y, y_hat):
		Return accuracy given y and y_hat 		