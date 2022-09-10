function total_rob_var=hiro_robust_variance(directions)
% compute the robust variance of the input set of directions
% directions is a matrix, with each of its column vector a found direction

global hat_t;
global initial_matrix;

                                           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
projected_value=directions'*initial_matrix; % projected_value is a matrix, %
                                           % while each column is a point
                                           % projected on the subspace given% 
                                           % by directions.
                                          % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
total_rob_var=0;
for j=1:size(projected_value, 1)
    temp=projected_value(j,:);
    for i=1:length(temp)
        temp(i)=temp(i)^2;
    end
    temp=sort(temp);                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                           %For each direction, list the
                                           %variance in an ascending order
                                           %%%%%%%%%%%%%%%%%%%%%%%%%%%55555
                                           
    total_rob_var=total_rob_var+sum(temp(1:hat_t)); 
end    
                                           


