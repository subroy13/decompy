function total_cor_matrix=calculate_correlation(observation_no)
% This function calculate the unnormalized correlation matrix for a given data_matrix.
% The output equals to correlation matrix times number of observations

global initial_matrix;
global data_remaining;

if sum(data_remaining)~= observation_no
     observation_no 
      error('the number of observations is not consistent with the columns of data_matrix');
end
if observation_no==size(initial_matrix, 2)
    total_cor_matrix=initial_matrix*initial_matrix';
else
    total_cor_matrix=initial_matrix*diag(data_remaining)*initial_matrix';
end    

%total_cor_matrix=zeros(size(initial_matrix,1));
%for i=1:length(data_remaining)
%    temp=initial_matrix(:,i);
%    total_cor_matrix=total_cor_matrix+temp*temp'*data_remaining(i);
%end    