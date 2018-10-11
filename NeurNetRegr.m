classdef NeurNetRegr < handle
  %NEURNETREGR neural network with fully connected layers
  % It a basic implementation for regression: the activation function is the
  % same for all hidden layer and linear for the output layer.
  %
  % INPUTS:
  % - input_layer_size: number of inputs (scalar)
  % - hidden_layer_sizes: number of neurons/layers (1d array)
  % - output_layer_size: number of outputs (scalar)
  %
  % USAGE:
  % nn = NeurNetRegr(input_layer_size, hidden_layer_sizes, output_layer_size)
  %
  % For training the NN, store the training set in the x_train and y_train properties
  %     nn.x_train = x_train; % size(x_train) == [no_examples, no_inputs]
  %     nn.y_train = y_train; % size(y_train) == [no_examples, no_outputs]
  %
  % then optimize the weights:
  %     cost = nn.optimize(n); % the number of iterations is optional
  %
  % to make predictions on an unseen test set:
  %     y_predicted = nn.predict(x_test)
  %
  % and to evaluate the accuracy:
  %     k = nn.accuracy(x_test, y_test)
  %
  % to reduce overfitting and improve the accuracy on the test set:
  %     nn.lambda = small_value;
  %
  % to debug the back-propagation algorithn:
  %     nn.checkGradients = true;
  %
  % to display supported activations and select one:
  %     disp(nn.all_activations);
  %     nn.activation_name = 'tanh';
  %
  % note that checkGradients works fine if the predicted value is close to the
  % true one, in that case their difference approximates well the derivative
  %
  % It is possible to find the inputs that matches a given ouput:
  %     [x_optimized, cost] = nn.optimizeInputs(x_first_guess, y_target)
  % This propagates back the error (y_predicted - y_target) from output to input,
  % so it works better if y_target is within the range of the outputs on which
  % the NN was trained on.
  
  % implementation notes:
  % the bias vectors are embedded in the weights matrices as first columns
  
  % 2018 Alberto Comin
  
  
  properties
    x_train; % train data-set (x_train)
    y_train; % train data-set (y_train)
    input_layer_size; % number of inputs
    output_layer_size = 1; % number of outputs
    hidden_layers_sizes = []; % array with number of neurons / layer
    lambda = 0; % L2 regularization factor
    weights = {}; % numel(weights) == numel(hidden_layers_sizes) + 1;
    cost; % cost calculated optimizing the NN on the train data-set
    checkGradients = false; % enable checking the back-prob gradient
    activation_name = 'tanh'; % name of activation function
  end
  
  properties(Dependent)
    all_activations; % names of supported activation functions
    activation; % handle to current activation function
    act_deriv; % handle to derivative of current activation function
  end
  
  properties(Access=protected)
    activations_dict; % dictionary with activation functions
  end
  
  properties(Dependent, Access=protected)
    nn_params; % all weights as one column array
  end
  
  methods
    function names = get.all_activations(obj)
      names = obj.activations_dict.keys;
    end
    
    function pms = get.nn_params(obj)
      % get all the weights as a single array
      pms = cellfun(@(x) reshape(x, 1, []), obj.weights, 'UniformOutput', false);
      pms = [pms{:}]';
    end
    
    function set.nn_params(obj, pms)
      % set the weights of the NN from a single array
      obj.weights = obj.unpackParams(pms);
    end
    
    function f = get.activation(obj)
      % get handle of current activation function
      f = obj.activations_dict(obj.activation_name).fun;
    end
    
    function f = get.act_deriv(obj)
      % get handle of derivative of current activation function
      f = obj.activations_dict(obj.activation_name).der;
    end
  end
  
  methods
    function obj = NeurNetRegr(input_layer_size, hidden_layer_sizes, output_layer_size)
      % construct aobject representing a neural network with fully-connected
      % layers
      if ~exist('output_layer_size', 'var'), output_layer_size = 1; end
      obj.input_layer_size    = input_layer_size;
      obj.output_layer_size   = output_layer_size;
      obj.hidden_layers_sizes = hidden_layer_sizes;
      
      obj.activations_dict = containers.Map( ...
        {'tanh', 'sigmoid', 'relu'}, { ...
        struct('fun', @tanh,                   'der', @(x) 1 - x.^2), ...
        struct('fun', @(x) 1 ./ (exp(-x) + 1), 'der', @(x) x - x.^2), ...
        struct('fun', @(x) max(0, x),          'der', @(x) double(x > 0)) ...
        });
      
      obj.initParams();
    end
    
    function initParams(obj)
      % initialize the params with random small values
      obj.weights = cell(numel(obj.hidden_layers_sizes) + 1, 1);
      obj.weights{1} = obj.randInitializeWeights( ...
        obj.input_layer_size, obj.hidden_layers_sizes(1));
      for i = 2 : numel(obj.hidden_layers_sizes)
        obj.weights{i} = obj.randInitializeWeights( ...
          obj.hidden_layers_sizes(i-1), obj.hidden_layers_sizes(i));
      end
      obj.weights{end} = obj.randInitializeWeights( ...
        obj.hidden_layers_sizes(end), obj.output_layer_size);
    end
    
    function step(obj, alpha)
      % do one step optimization on the train set
      if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1e-2; % coefficient for gradient descent
      end
      [~, grad, ~] = obj.nnCostFunction(obj.nn_params, ...
        obj.x_train, obj.y_train, obj.lambda);
      obj.nn_params = obj.nn_params - alpha * grad;
    end
    %%
    function J = optimize(obj, maxIter)
      % optimize the NN on the train set and return the cost
      if ~exist('maxIter','var') || isempty(maxIter)
        maxIter = 100;
      end
      
      costFunction = @(p) obj.nnCostFunction(p, obj.x_train, obj.y_train, obj.lambda);
      
      options_fmincg = optimset('MaxIter', maxIter);
      % fmincg is faster that the default matlab fminunc
      [pms, J] = fmincg(costFunction, obj.nn_params, options_fmincg);
      obj.nn_params = pms;
      obj.cost = J;
    end
    %%
    function [X2, J] = optimizeInputs(obj, X1, y1, lb, ub, maxIter, outputNo)
      % optimize the inputs to match a specified output
      if ~exist('maxIter','var') || isempty(maxIter)
        maxIter = 100;
      end
      if ~exist('lb','var') || isempty(lb)
        lb = -inf(length(X1), 1);
      end
      if ~exist('ub','var') || isempty(ub)
        ub = inf(length(X1), 1);
      end
      if ~exist('outputNo', 'var') || isempty(outputNo)
        assert(size(obj.y_train, 2) == size(y1, 2), ...
          'NeureNetRegr:optimizeInputs:argChk', ...
          'size(y_train, 2) should be equal to number of output of neural net');
        mask = true(1, size(y1,2));
      else
        assert(size(y1, 2) == length(outputNo), ...
          'NeureNetRegr:optimizeInputs:argChk', ...
          'mask should contain the indeces of the output columns provided');
        mask = false(1, size(obj.y_train,2));
        mask(outputNo) = true;
        temp = nan(size(y1, 1), size(obj.y_train, 2));
        temp(:, outputNo) = y1;
        y1 = temp;
      end
      
      ws = obj.nn_params; % current weights in one single array
      costFunction = @(t) obj.inputCostFunction(ws, t, y1, mask);
      
      options = optimoptions(@fmincon, ...
        'Display',                 'final', ...
        'Algorithm',               'trust-region-reflective', ...
        'SpecifyObjectiveGradient', true, ...
        'CheckGradients',           obj.checkGradients, ...
        'OptimalityTolerance',      1e-20, ...
        'FunctionTolerance',        1e-10, ...
        'OptimalityTolerance',      1e-20, ...
        'MaxIterations',            maxIter, ...
        'PlotFcn',                  {'optimplotfval','optimplotfirstorderopt'});
      [X2, J] = fmincon(costFunction, X1(:), [],[], [],[], lb, ub, [], options);
      X2 = reshape(X2, size(X1));
    end
    %%
    function p = predict(obj, X)
      % predict the output values of a specified test set
      if ~exist('X', 'var')
        X = obj.x_train;
      end
      m = size(X, 1);
      
      A = cell(numel(obj.weights), 1);
      A{1} = [ones(m,1), X];
      for i = 2 : numel(A)
        A{i} = [ones(m,1), obj.activation(A{i - 1} * obj.weights{i - 1}')];
      end
      p = (A{end} * obj.weights{end}');
    end
    
    function k = accuracy(obj, X, y)
      % calculates the prediction accuracy on a test set
      if ~exist('X', 'var') || isempty(X)
        X = obj.x_train;
      end
      if ~exist('y', 'var') || isempty(y)
        y = obj.y_train;
      end
      p = obj.predict(X);
      k = sqrt(mean((p(:) - y(:)).^2));
    end
    %%
    function [J, D0] = inputCostFunction(obj, ws, xi, ym, mask)
      % Calculate the cost function and its gradient for input optimization.
      % The mask allows to optimize on a subset of output values.
      % This is just a wraper on nnCostFunction, because we need the
      % input gradient as second output for the optimization functions.
      
      if ~exist('mask', 'var') || isempty(mask)
        mask = true(1, size(ym, 2));
      end
      
      X = reshape(xi, size(ym, 1), obj.input_layer_size);
      [J, ~, D0] = obj.nnCostFunction(ws, X, ym, 0, mask);
      D0 = reshape(D0, size(xi));
    end
    %%
    function [J, grad, input_grad] = nnCostFunction(obj, pms, X, ym, lambda, mask)
      % Calculate cost function, gradiend and input gradient
      % the mask allowss to optimize on a subset of output values.
      % Note that the packing/unpacking of the weights into a single array is
      % needed because matlab optimization functions require a single array but
      % for the calculations we need the individual matrices.
      
      if ~exist('mask', 'var') || isempty(mask)
        mask = true(1, size(ym, 2));
      end
      
      % initialize the weights
      ws = obj.unpackParams(pms);
      
      % the forward pass
      m = size(X, 1);
      A = cell(numel(ws), 1);
      A{1} = [ones(m, 1), X];
      for i = 2 : numel(A)
        A{i} = [ones(m,1), obj.activation(A{i-1} * ws{i-1}')];
      end
      pred = (A{end} * ws{end}');
      
      % the cost function including L2 regularization (sums of the squares of
      % the weights, excuding the first-column because it is the bias)
      J = 1 / 2 / m * ( sum(sum((ym(:, mask) - pred(:, mask)).^2)) + ...
        lambda * sum(cellfun(@(x) sum(sum(x(:, 2:end).^2)), ws)) );
      
      % numel(weights) == numel(hidden_layers_sizes) + 1
      D0 = zeros(size(X));
      D = cellfun(@(x) zeros(size(x)), ws, 'UniformOutput', false);
      
      % iterate over the test cases, sum up the error on the outputs,
      % back-propagate the error to th input
      deriv = obj.act_deriv; % in matlab this indirection is expensive
      for t = 1:m
        y = ym(t, :)';
        % the ouput error is the difference between predicted and true values
        err = zeros(size(y));
        err(mask) = (pred(t, mask)' - y(mask)) ;
        % the backward pass
        d = cell(numel(ws), 1);
        d{end} = (ws{end}' * err) .* deriv(A{end}(t, :)');
        for i = numel(ws) - 1 : -1 : 2
          d{i} = (ws{i}' * d{i+1}(2:end)) .* deriv(A{i}(t, :)');
        end
        d{1} = (ws{1}' * d{2}(2:end)) ;
        D{end} = D{end} + err * A{end}(t, :);
        for i = numel(ws) - 1 : -1 : 1
          D{i} = D{i} + d{i+1}(2:end) * A{i}(t, :);
        end
        D0(t,:) = d{1}(2:end);
      end
      
      % normalize by the number of test cases and add the regularization term
      input_grad  = 1/m * D0;
      for i = 1 : numel(D)
        D{i}(:, 2:end) = D{i}(:, 2:end) + lambda * ws{i}(:, 2:end);
        D{i} = D{i} / m;
      end
      
      % reshape the gradient as a single array, needed by optimization functions
      grad = cellfun(@(x) reshape(x, 1, []), D, 'UniformOutput', false);
      grad = reshape([grad{:}], size(pms));
      
    end
    
    function weights = unpackParams(obj, pms)
      % unpack the single weights matrices from a single array
      
      n = nan(numel(obj.hidden_layers_sizes) + 1, 1);
      n(1) = obj.hidden_layers_sizes(1) * (obj.input_layer_size + 1);
      
      for i = 2 : numel(n) - 1
        n(i) = obj.hidden_layers_sizes(i) * (obj.hidden_layers_sizes(i-1) + 1);
      end
      n(end) = obj.output_layer_size * (obj.hidden_layers_sizes(end) + 1);
      
      cumul_sizes = cumsum(n);
      weights = cell(numel(obj.hidden_layers_sizes) + 1, 1);
      
      weights{1} = reshape( pms(1 : n(1)), obj.hidden_layers_sizes(1), ...
        obj.input_layer_size + 1);
      
      for i = 2 : numel(weights) - 1
        weights{i} = reshape(pms(cumul_sizes(i-1) + 1 : cumul_sizes(i)), ...
          obj.hidden_layers_sizes(i), obj.hidden_layers_sizes(i-1) + 1);
      end
      
      weights{end} = reshape(pms(cumul_sizes(end-1) + 1 : cumul_sizes(end)), ...
        obj.output_layer_size, obj.hidden_layers_sizes(end) + 1);
    end
  end
  
  methods(Static, Access=protected)
    function W = randInitializeWeights(L_in, L_out, eps_init)
      % initialize the weights of one layer with small random numbers
      if ~exist('eps_init', 'var')
        eps_init = 1e-2;
      end
      W = 2 * eps_init * rand(L_out, 1 + L_in) - eps_init;
    end
  end
end

