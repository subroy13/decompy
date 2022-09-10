function output=hiro_random_removal(directions)
% randomly remove a point based on the total variance on the projected
% directions
% directions is a matrix, with each of its column vector a found direction

global data_remaining;
global initial_matrix;
global fake_zero;

projected_value=directions'*initial_matrix;

total_variance=0;
for i=1:size(projected_value, 2)
    projected_variance(i)=0;
    for j=1:size(projected_value, 1)
       projected_variance(i)=projected_variance(i)+projected_value(j,i)^2*data_remaining(i);
    end   
    total_variance=total_variance+projected_variance(i);
end    

temp_rand=total_variance*rand(1);
indicator=0;

%removing a point based on the variance
for i=1:size(projected_value,2)
   temp_rand=temp_rand-projected_variance(i);
   if temp_rand<= fake_zero
       output=initial_matrix(:,i);
       data_remaining(i)=0;
       indicator=1;
       break;
   end
end   

%removing the last point when no point is removed. However, this is due to
%some numerical problems.
if indicator==0
    warning('numerical problem in random-removal, the last sample removed')
    for i=length(data_remaining):-1:1
        if data_remaining==1
            output=initial_matrix(:,i);
              data_remaining(i)=0;
              break;
        end
    end   
end    

