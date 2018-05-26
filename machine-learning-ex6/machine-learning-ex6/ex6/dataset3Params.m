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

tens = 10 .^ (-2:2);
baseFactors = 1:2:3;
[t, b] = meshgrid(tens, baseFactors);
sigma_predicts = (t(:) .* b(:))';
C_predicts = sigma_predicts;

predictions_error = zeros(length(C_predicts), length(sigma_predicts));

for i=1:length(C_predicts)
    for j=1:length(sigma_predicts)
        C = C_predicts(i);
        sigma = sigma_predicts(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        predictions_error(i,j) = mean(double(predictions ~= yval));
    end
end

minimum_error = min(min(predictions_error));
[i, j] = find(predictions_error == minimum_error);
C = C_predicts(i)
sigma = sigma_predicts(j)

% =========================================================================

end
