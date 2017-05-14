function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% X is 5000 x 400 matrix
% y is 5000 x 1 vector

%size(X)

%size(y)


% First Convert y to matrix with each row element if y will be mapped to 
%a binary (0 or 1) based on the total number of unique values in y

number_of_unique_values = length(unique(y)); %number_of_unique_values = no of classes
Y = zeros(number_of_unique_values, m);
for i = 1:m
    Y(y(i), i) = 1;
endfor

% now Y is 10 x 5000 matrix
%size(Y)

% Part 1
%==========================

% Add ones to the X data matrix
X = [ones(m, 1) X];
a1 = X;

%fprintf('Size of a1\n');
%size(X)


%theta1 is 25x401 matrix
%X or a1 is 5000x401 matrix;
%a1 = a1' to make matrix multiplication compatible
a1 = a1';
z2 = Theta1 * a1;

a2 = sigmoid(z2);

%fprintf('Size of a2\n');
%size(a2)
% a2 is 25x5000 matrix
%theta2 is 10x26 matrix

a2 = [ones(1,m); a2];

% now a2 is 26x5000 matrix
z3 = Theta2 * a2;
a3 = sigmoid(z3);

h = a3;

% computing cost function
J = (1/m)*sum(sum(-Y.*log(h) - (1-Y).*log(1-h)));


% J with regularization parameters`

%%theta1_without_bias = theta1(:,2:end);

%%theta2_without_bias = theta2(:,2:end);

%%penalize = sum(sum(theta1_without_bias .^ 2)) + ...
%%				sum(sum(theta2_without_bias .^ 2));

penalize = sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2));
regularization_parameter = (lambda/(2*m)) * penalize;

J = J + regularization_parameter;

%%J = J + (lambda/(2*m)) * penalize;



% Part 2
%=============================

delta_3 = a3 - Y;

%delta_2_temp = Theta2'*delta_3; % .* sigmoidGradient(z2);
%fprintf('Size of delta_2_temp \n');
%size(delta_2_temp)

%%sig = sigmoidGradient(z2);

%fprintf('Size of sig \n');
%size(sig)

%to remove the bias(1st column) as I was getting a calculation error
%when I did not remove the bias.
% both martices on left and right of multiplication sign is 25 x 5000
delta_2 = (Theta2'*delta_3)(2:end,:) .* sigmoidGradient(z2);

%%delta_2 = delta_2(2:end);


% Unregularized gradient calculation
theta_2_unreg_grad = (delta_3 * a2')/m;
thata_1_unreg_grad = (delta_2 * a1')/m;	


%Theta1_unreg_grad = (delta_2 * A1')/m;
%Theta2_unreg_grad = (delta_3 * A2')/m;

% Regularize
Theta1_grad = thata_1_unreg_grad + (lambda/m) * Theta1;
Theta2_grad = theta_2_unreg_grad + (lambda/m) * Theta2;

Theta1_grad(:, 1) = thata_1_unreg_grad(:, 1);
Theta2_grad(:, 1) = theta_2_unreg_grad(:, 1);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
