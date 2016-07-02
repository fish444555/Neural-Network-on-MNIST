function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%% NN
X = [ones(m,1) X];
a2_1 = X;
z2_2 = a2_1*Theta1';
a2_2 = sigmoid(z2_2);
a2_2 = [ones(size(a2_2,1),1) a2_2];
z2_3 = a2_2*Theta2';
a2_3 = sigmoid(z2_3);

for i = 1:m
    for j = 1:size(a2_3,2)
        if(j==y(i))
            J = J-log(a2_3(i,j));
        else
            J = J-log(1-a2_3(i,j));
        end
    end
end
J = J/m;
J = J+lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/(2*m);

yr = zeros(m,num_labels);
for i = 1:m
    if y(i) == 0
        yr(i, 10) = 1;
        continue;
    end
    yr(i,y(i)) = 1;
end

delta3 = a2_3 - yr;

tempd2 = delta3*Theta2;
delta2 = tempd2(:,2:end).*sigmoidGradient(z2_2);

Theta2_grad = Theta2_grad+delta3'*a2_2;
Theta1_grad = Theta1_grad+delta2'*a2_1;

Theta2_grad = Theta2_grad/m+lambda*[zeros(size(Theta2,1),1),Theta2(:,2:end)]/m;
Theta1_grad = Theta1_grad/m+lambda*[zeros(size(Theta1,1),1),Theta1(:,2:end)]/m;

%% NN end

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


