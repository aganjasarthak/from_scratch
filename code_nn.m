

function [W, b, train_loss_history, test_loss_history] = ....
   
    input_size = size(X_train_poly, 2);
    hidden_sizes = repmat(128, 1, 10); % 10 hidden layers with 128 neurons each
    output_size = 1;
    num_layers = length(hidden_sizes) + 1;

    
    W = cell(1, num_layers);
    b = cell(1, num_layers);
    for i = 1:num_layers
        if i == 1
            W{i} = randn(input_size, hidden_sizes(i)) * sqrt(2 / (input_size + hidden_sizes(i)));
            b{i} = zeros(1, hidden_sizes(i));
        elseif i == num_layers
            W{i} = randn(hidden_sizes(end), output_size) * sqrt(2 / (hidden_sizes(end) + output_size));
            b{i} = zeros(1, output_size);
        else
            W{i} = randn(hidden_sizes(i-1), hidden_sizes(i)) * sqrt(2 / (hidden_sizes(i-1) + hidden_sizes(i)));
            b{i} = zeros(1, hidden_sizes(i));
        end
    end

   
    iterations = 5000;
    batch_size = min(64, size(X_train_poly, 1));
    learning_rate = 0.01;
    lambda = 0.00123;  % L2
    activation_functions = {'relu', 'leaky_relu'}; 

    % Adam 
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

   
    m_W = cell(1, num_layers);
    v_W = cell(1, num_layers);
    m_b = cell(1, num_layers);
    v_b = cell(1, num_layers);
    for i = 1:num_layers
        m_W{i} = zeros(size(W{i}));
        v_W{i} = zeros(size(W{i}));
        m_b{i} = zeros(size(b{i}));
        v_b{i} = zeros(size(b{i}));
    end

 
    train_loss_history = zeros(1, iterations);
    test_loss_history = zeros(1, iterations);

   
    for iter = 1:iterations
     
        batch_indices = randperm(size(X_train_poly, 1), batch_size);
        X_batch = X_train_poly(batch_indices, :);
        y_batch = y_train(batch_indices);
        
        
        a = cell(1, num_layers+1);
        z = cell(1, num_layers);
        a{1} = X_batch;
        
        for i = 1:num_layers
          
            z{i} = a{i} * W{i};
            if ~isempty(b{i})
                z{i} = z{i} + repmat(b{i}, size(a{i}, 1), 1);
            end
            
            if i == num_layers
                a{i+1} = z{i};  
            else
               
                switch activation_functions{mod(i-1, length(activation_functions)) + 1}
                    case 'relu'
                        a{i+1} = max(0, z{i});
                    case 'leaky_relu'
                        a{i+1} = max(0.01 * z{i}, z{i});
                end
            end
        end
        y_pred = a{end};
        
       
        train_loss = mean((y_pred - y_batch).^2) + lambda * sum(cellfun(@(w) sum(w(:).^2), W));
        train_loss_history(iter) = train_loss;

      
        a_test = cell(1, num_layers+1);
        a_test{1} = X_test_poly;
        for i = 1:num_layers
            z_test = a_test{i} * W{i};
            if ~isempty(b{i})
                z_test = z_test + repmat(b{i}, size(a_test{i}, 1), 1);
            end
            
            if i == num_layers
                a_test{i+1} = z_test;
            else
                switch activation_functions{mod(i-1, length(activation_functions)) + 1}
                    case 'relu'
                        a_test{i+1} = max(0, z_test);
                    case 'leaky_relu'
                        a_test{i+1} = max(0.01 * z_test, z_test);
                end
            end
        end
        y_pred_test = a_test{end};
        test_loss = mean((y_pred_test - y_test).^2);
        test_loss_history(iter) = test_loss;

       
        delta = cell(1, num_layers);
        delta{num_layers} = 2 * (y_pred - y_batch) / batch_size; % Loss derivative
        
        for i = num_layers-1:-1:1
            
            grad = delta{i+1} * W{i+1}';
            grad = grad(:, 1:size(z{i}, 2)); 
            
            switch activation_functions{mod(i-1, length(activation_functions)) + 1}
                case 'relu'
                    delta{i} = grad .* (z{i} > 0);
                case 'leaky_relu'
                    delta{i} = grad .* ((z{i} > 0) + 0.01 * (z{i} <= 0));
            end
        end
        
       
        
        dW = cell(1, num_layers);
        db = cell(1, num_layers);
        for i = 1:num_layers
            dW{i} = a{i}' * delta{i} + 2 * lambda * W{i};
            db{i} = sum(delta{i}, 1);
        end
        
    
        
        for i = 1:num_layers
           
            m_W{i} = beta1 * m_W{i} + (1 - beta1) * dW{i};
            v_W{i} = beta2 * v_W{i} + (1 - beta2) * (dW{i}.^2);
            m_W_hat = m_W{i} / (1 - beta1^iter);
            v_W_hat = v_W{i} / (1 - beta2^iter);
            W{i} = W{i} - learning_rate * m_W_hat ./ (sqrt(v_W_hat) + epsilon);
            
           
            m_b{i} = beta1 * m_b{i} + (1 - beta1) * db{i};
            v_b{i} = beta2 * v_b{i} + (1 - beta2) * (db{i}.^2);
            m_b_hat = m_b{i} / (1 - beta1^iter);
            v_b_hat = v_b{i} / (1 - beta2^iter);
            b{i} = b{i} - learning_rate * m_b_hat ./ (sqrt(v_b_hat) + epsilon);
        end
      
     
        if mod(iter, 500) == 0
            learning_rate = learning_rate * 0.85; 
        end
        
     
        if mod(iter, 5) == 0
            fprintf('Iteration %d: Train Loss = %.6f, Test Loss = %.6f\n', iter, train_loss, test_loss);
        end
       
        if train_loss < 0.01
            fprintf('Converged at iteration %d\n', iter);
            break;
        end
    end
end
