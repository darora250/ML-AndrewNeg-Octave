function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
a1 = X;

fprintf('Size of a1\n');
size(X)


%theta1 is 25x401 matrix
%X or a1 is 5000x401 matrix;
z2 = Theta1 * a1';

a2 = sigmoid(z2);

fprintf('Size of a2\n');
size(a2)
% a2 is 25x5000 matrix
%theta2 is 10x26 matrix

a2 = [ones(1,m); a2];

% now a2 is 26x5000 matrix
z3 = Theta2 * a2;
a3 = sigmoid(z3);

%a3 is 10x5000
fprintf('Size of a3\n');
size(a3)

% by using max(a3,[],1), the resulting p is row vector. But, the 
%training accuracy is calculated (line 69 in ex3_nn.m) w.r.t. y 
% which is a column vector. Since p should also be returned as a column 
%vector, either transpose p before returning from this method or use
% max(a3',[],2)

[value, p] = max(a3,[],1);
p = p'; %% to conver p as a column vector

%[value, p] = max(a3',[],2);



%Another straightforward approach
%
%A1 = [ones(1, m); X'];
%A2 = [ones(1, m); sigmoid(Theta1*A1)];
%A3 = sigmoid(Theta2*A2);
%[value, p] = max(A3', [], 2);
%




% =========================================================================


end
