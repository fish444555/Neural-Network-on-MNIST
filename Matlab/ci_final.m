clear ; close all; clc


fprintf('Loading and Visualizing Data ...\n')

num_labels = 10;          % 10 labels, from 1 to 10   

small_data_set = 1;
if(small_data_set == 1)
    load('digit_data.mat');
    m = size(X, 1);
    input_layer_size  = size(X,2);  % 20x20 Input Images of Digits
%     hidden_layer_size = 25;   % 25 hidden units
    sep_rate = 0.7;
    mtrain = round(m * sep_rate);
    mtest = m - mtrain;
    permute_idx = randperm(m);
    Xp = X(permute_idx, :);
    Yp = y(permute_idx, :);
    X = Xp(1: mtrain, :);
    y = Yp(1: mtrain, :);
    m = mtrain;
    X_test = Xp(1: mtest, :);
    y_test = Yp(1: mtest, :);
    sel = randperm(size(X, 1));
    sel = sel(1:100);
    displayData(X(sel, :));
else
    X = loadMNISTImages('train-images-idx3-ubyte');
    X = X';
    y = loadMNISTLabels('train-labels-idx1-ubyte');
    y = y';
    m = size(X, 1);
    input_layer_size  = size(X,2);  % 20x20 Input Images of Digits
    hidden_layer_size = 25;   % 25 hidden units
end




fprintf('\nTraining Neural Network... \n')


grid_search = 0;
if(grid_search == 1)
%     hidden_layer_size = [10, 25];
    hidden_layer_size = [10, 15, 25, 50, 100];
% hidden_layer_size = [10, 25];
%     lambda = [0, 0.1, 0.5, 0.8, 1, 1.5];
    lambda = [0, 1];
else
    hidden_layer_size = 25;
    lambda = 1;
    costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

end
options = optimset('MaxIter', 100);




if(grid_search == 1)
    iter_num = [30, 50, 100, 150, 200];    
    average_num = 5;
    average_rec = zeros(1, average_num);
    grid_rec = zeros(size(iter_num, 2), size(hidden_layer_size, 2), size(lambda, 2));
    for idx_iter = 1: size(iter_num, 2)
        options = optimset('MaxIter', iter_num(idx_iter));
        for i = 1: size(hidden_layer_size, 2)
            for j = 1: size(lambda, 2)
                for k = 1: average_num
                    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size(i));
                    initial_Theta2 = randInitializeWeights(hidden_layer_size(i), num_labels);
                    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

                    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size(i), ...
                                       num_labels, X, y, lambda(j));


                    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

                    Theta1 = reshape(nn_params(1:hidden_layer_size(i) * (input_layer_size + 1)), ...
                                     hidden_layer_size(i), (input_layer_size + 1));

                    Theta2 = reshape(nn_params((1 + (hidden_layer_size(i) * (input_layer_size + 1))):end), ...
                                     num_labels, (hidden_layer_size(i) + 1));


                    if(small_data_set == 1)
                        pred = predict(Theta1, Theta2, X_test);
                        fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);
                    else
                        X_test = loadMNISTImages('t10k-images-idx3-ubyte');
                        X_test = X_test';
                        y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
                        pred = predict(Theta1, Theta2, X_test);
                        fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);
                    end
                    if small_data_set  == 1
                        average_rec(1, k) = mean(double(pred == y_test)) * 100;
                    else
                        average_rec(1, k) = mean(double(pred == y_test)) * 100;
                    end
                end
                grid_rec(idx_iter, i, j) = mean(average_rec);
            end
        end
    end
else
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));


    if(small_data_set == 1)
        pred = predict(Theta1, Theta2, X_test);
        fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);
    else
        X_test = loadMNISTImages('t10k-images-idx3-ubyte');
        X_test = X_test';
        y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
        pred = predict(Theta1, Theta2, X_test);
        fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);
    end
end