Classification Class Read Me-------------------

init - ANN_Classification(self,N,Fs) --

	N[]: list of number of nodes in for each layer - defult 4 nodes
	Fs[]: list active function for each layer - defult 1 layer of ReLU


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

	creatWbs(self)
		create defult W and B

	creat_DFs(R)
		create functions for back propagation

	feed_forward(self,x)
		for word back propagation return list of Z and P 

			