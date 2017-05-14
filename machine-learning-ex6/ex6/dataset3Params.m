function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;



% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_values = [0.01 0.03 0.1 0.3 1 3 10 30]'; %'
sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30]'; %'

result_matrix = [];

for i = 1:length(C_values)
	for j = 1:length(sigma_values)

		%%calculate the model
		model= svmTrain(X, y, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma_values(j)));

		%%get prediction for cross validation set Xval
		predictions = svmPredict(model, Xval);

		%%calculate error for model trained by training set and validated by cross-validation set
		error = mean(double(predictions ~= yval));

		%%put all the values in a matrix in the form [error C sigma]
		result_matrix = [result_matrix; error C_values(i) sigma_values(j)];
		
	end
end

result_matrix_sorted_by_first_column = sortrows(result_matrix);
row_with_minimum_error = result_matrix_sorted_by_first_column(1,:);

C = row_with_minimum_error(2);

sigma = row_with_minimum_error(3);

fprintf(['\n C = %f , sigma = %f , error = %f '], C, sigma, row_with_minimum_error(1));








% =========================================================================

end
