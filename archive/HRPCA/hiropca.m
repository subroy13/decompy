function  opt_direction=hiropca(input_data, varargin)
%This function is the hiropca algorithm
% Input includes 'pc_no', 'lambda', 'T', 'hat_t'
% Default T is half of the sample_no
global initial_matrix;
global data_remaining;
global fake_zero;
global hat_t;

initial_matrix=input_data;
clear input_data;

observation_no=size(initial_matrix, 2);
data_remaining=ones(1, observation_no);


T=floor(observation_no/2);
hat_t_ind=-1;
lambda=0.1;

if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;
for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'pc_no'
            input_d=parameterValue;
        case 'lambda'
            lambda=parameterValue;
        case 'hat_t'
            hat_t_ind=parameteValue;
        case 't'
            T=parameterValue;
 otherwise
            error(['Sorry, the parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end

sample_no=floor(observation_no*(1-lambda));

if hat_t_ind == -1
   % if lambda<0.1
      if lambda==0
        hat_t=floor(0.9*sample_no);
    else 
        hat_t=sample_no;
    end
else
    hat_t=hat_t_ind;
end    



fake_zero=0.000000001;

cov_matrix=calculate_correlation(observation_no);

OPT.disp=0;
OPT.maxit=3000;

opt_value=0;
for i=1:T
    i
    [directions, eigen_temp]=eigs(cov_matrix, input_d, 'LM', OPT);
    temp=hiro_robust_variance(directions);
    if temp>opt_value
        opt_value=temp;
        opt_direction=directions;
        opt_no=i;
    end    
    removed_vector=hiro_random_removal(directions);
    cov_matrix=cov_matrix-removed_vector*removed_vector';
end    
opt_value
opt_no