% Task: Demonstrate the worst-case prior is not necessarily equivalent to
%        worst model fitting
%
%  Four priors: 1. uniform prior (non-informative prior)
%               2. EB prior (estimated by empirical Bayes)
%               3. Informative prior 1
%               4. Informative prior 2
%
%  Simple MDP: Three states and two action
%              s0: p(s1|s0,a),p(s2|s0,a);p(s1|s0,b),p(s2|s0,b)
%              s1: absorbing state, R(s1)=1;
%              s2: absorbing state, R(s2)=1;
%
%  Beta-nominal model

clear
rng(1);
%%

%%
% Parameter Settings
n_trials_list = 8;

p_a = [0.45, 0.55]; % probability under action a; true optimal action is b, true optimal value is 0.55;
p_b = [0.5, 0.5]; % probability under action b;

reward_a = [1 0];
reward_b = [1 0];
value = [p_a*reward_a';p_b*reward_b'];

[value_opt,action_opt] = max(value);
fixed_utility = value(2);


% Fixed Informative Prior parameters (globally consistent)

rng(1);
K = 15;
k1 = 1:2:K-1;
thre_list = [1e-6,1e-4,1e-3,1e-3,1e-3,1e-4];


experiment_idx = 3;  % 1,2,..., 6

thre = thre_list(experiment_idx);


% Parameters for the first informative prior
alpha_info_global1 = k1(experiment_idx);
beta_info_global1 = K-k1(experiment_idx);

% Added parameters for the second informative prior
alpha_info_global2 = K-k1(experiment_idx);
beta_info_global2 = k1(experiment_idx);

% % Parameters for the first informative prior
% alpha_info_global1 = 5;
% beta_info_global1 = 10;
% 
% % Added parameters for the second informative prior - biased in the other direction
% alpha_info_global2 = 10;
% beta_info_global2 = 5;

% Initialize storage structure
results = struct();

for n_idx = 1:length(n_trials_list)
    n_data = n_trials_list(n_idx);
    y_values = 0:n_data;
    
    % Preallocate arrays - added variables for informative2
    loglik_unif = zeros(size(y_values));
    loglik_eb = zeros(size(y_values));
    loglik_rand1 = zeros(size(y_values));
    loglik_rand2 = zeros(size(y_values)); % New
    utility_unif = zeros(size(y_values));
    utility_eb = zeros(size(y_values));
    utility_rand1 = zeros(size(y_values));
    utility_rand2 = zeros(size(y_values)); % New
    max_unif = zeros(size(y_values));
    max_eb = zeros(size(y_values));
    max_rand1 = zeros(size(y_values));
    max_rand2 = zeros(size(y_values)); % New
    min_utility = zeros(size(y_values));
    out_sample_unif = zeros(size(y_values));
    out_sample_eb = zeros(size(y_values));
    out_sample_rand1 = zeros(size(y_values));
    out_sample_rand2 = zeros(size(y_values)); % New
    out_sample_robust = zeros(size(y_values));
    min_prior_type = cell(size(y_values)); % Store the type of prior that minimizes utility
    
    for y_idx = 1:length(y_values)
        y_data = y_values(y_idx);
        
        % ===== Basic Calculations =====
        % --- Uniform Prior ---
        alpha = 1; beta = 1;
        loglik_unif(y_idx) = beta_binomial_marginal_loglikelihood(y_data, n_data, alpha, beta);
        p_predict = (y_data + alpha) / (n_data + alpha + beta);
        utility_unif(y_idx) = sum(reward_a .* [p_predict, 1-p_predict]);
        
        % --- Empirical Bayes Prior ---
        try
            [alpha_eb, beta_eb] = empirical_bayes_beta_binomial(y_data, n_data);
            loglik_eb(y_idx) = beta_binomial_marginal_loglikelihood(y_data, n_data, alpha_eb, beta_eb);
            p_predict = (y_data + alpha_eb) / (n_data + alpha_eb + beta_eb);
            utility_eb(y_idx) = sum(reward_a .* [p_predict, 1-p_predict]);
        catch
            loglik_eb(y_idx) = NaN;
            utility_eb(y_idx) = NaN;
        end
        
        % --- Informative Prior 1 ---
        loglik_rand1(y_idx) = beta_binomial_marginal_loglikelihood(y_data, n_data, alpha_info_global1, beta_info_global1);
        p_predict = (y_data + alpha_info_global1) / (n_data + alpha_info_global1 + beta_info_global1);
        utility_rand1(y_idx) = sum(reward_a .* [p_predict, 1-p_predict]);
        
        % --- Informative Prior 2 (New) ---
        loglik_rand2(y_idx) = beta_binomial_marginal_loglikelihood(y_data, n_data, alpha_info_global2, beta_info_global2);
        p_predict = (y_data + alpha_info_global2) / (n_data + alpha_info_global2 + beta_info_global2);
        utility_rand2(y_idx) = sum(reward_a .* [p_predict, 1-p_predict]);
        
        % ===== Task Calculations =====
        % Task 1: Compare each prior's utility with the fixed utility
        % --- Uniform Prior ---
        if abs(utility_unif(y_idx) - fixed_utility) < 1e-6  % Note precision
            idx = action_opt; % Choose optimal action in case of tie
            max_unif(y_idx) = max([utility_unif(y_idx), fixed_utility]);
        else
            [max_unif(y_idx), idx] = max([utility_unif(y_idx), fixed_utility]);
        end
        out_sample_unif(y_idx) = value(idx);  % out-of-sample value

        % --- Empirical Bayes Prior ---
        if abs(utility_eb(y_idx) - fixed_utility) < 1e-6  % Note precision
            idx = action_opt; % Choose optimal action in case of tie
            max_eb(y_idx) = max([utility_eb(y_idx), fixed_utility]);
        else
            [max_eb(y_idx), idx] = max([utility_eb(y_idx), fixed_utility]);
        end
        out_sample_eb(y_idx) = value(idx);    % out-of-sample value

        % --- Random Prior 1 ---
        if abs(utility_rand1(y_idx) - fixed_utility) < 1e-6  % Note precision
            idx = action_opt; % Choose optimal action in case of tie
            max_rand1(y_idx) = max([utility_rand1(y_idx), fixed_utility]);
        else
            [max_rand1(y_idx), idx] = max([utility_rand1(y_idx), fixed_utility]);
        end
        out_sample_rand1(y_idx) = value(idx);  % out-of-sample value

        % --- Random Prior 2 ---
        if abs(utility_rand2(y_idx) - fixed_utility) < 1e-6  % Note precision
            idx = action_opt; % Choose optimal action in case of tie
            max_rand2(y_idx) = max([utility_rand2(y_idx), fixed_utility]);
        else
            [max_rand2(y_idx), idx] = max([utility_rand2(y_idx), fixed_utility]);
        end
        out_sample_rand2(y_idx) = value(idx);  % out-of-sample value
        
        % Task 2: Only compare priors with log_likelihood >= loglik_eb + log(thre)
        valid_prior_mask = false(1,4); % [Non-informative, eb, informative1, informative2]
        loglik_threshold = loglik_eb(y_idx) + log(thre);
        
        % Check if each prior satisfies the condition
        if loglik_unif(y_idx) >= loglik_threshold
            valid_prior_mask(1) = true;
        end
        if ~isnan(loglik_eb(y_idx)) % EB always participates in comparison
            valid_prior_mask(2) = true;
        end
        if loglik_rand1(y_idx) >= loglik_threshold
            valid_prior_mask(3) = true;
        end
        if loglik_rand2(y_idx) >= loglik_threshold
            valid_prior_mask(4) = true;
        end
        
        % Collect utilities that satisfy the condition and their types
        valid_utilities = [];
        prior_types = {};
        if valid_prior_mask(1)
            valid_utilities = [valid_utilities, utility_unif(y_idx)];
            prior_types{end+1} = 'Non-informative';
        end
        if valid_prior_mask(2)
            valid_utilities = [valid_utilities, utility_eb(y_idx)];
            prior_types{end+1} = 'EB';
        end
        if valid_prior_mask(3)
            valid_utilities = [valid_utilities, utility_rand1(y_idx)];
            prior_types{end+1} = 'Informative1';
        end
        if valid_prior_mask(4)
            valid_utilities = [valid_utilities, utility_rand2(y_idx)];
            prior_types{end+1} = 'Informative2';
        end
        
        % Calculate the minimum and compare with fixed utility, and record the minimizing prior type
        if isempty(valid_utilities)
            min_utility(y_idx) = fixed_utility;
            min_prior_type{y_idx} = 'None';
        else
            [min_val, min_idx] = min(valid_utilities);
            [min_utility(y_idx),idx] = max([min_val, fixed_utility]);
            out_sample_robust(y_idx) = value(idx);  % out-of-sample value
            min_prior_type{y_idx} = prior_types{min_idx};
        end
    end
    
    % Store results - added variables for informative2
    results(n_idx).n_data = n_data;
    results(n_idx).y_values = y_values;
    results(n_idx).loglik_unif = loglik_unif;
    results(n_idx).loglik_eb = loglik_eb;
    results(n_idx).loglik_rand1 = loglik_rand1;
    results(n_idx).loglik_rand2 = loglik_rand2;
    results(n_idx).utility_unif = utility_unif;
    results(n_idx).utility_eb = utility_eb;
    results(n_idx).utility_rand1 = utility_rand1;
    results(n_idx).utility_rand2 = utility_rand2;
    results(n_idx).max_unif = max_unif;
    results(n_idx).max_eb = max_eb;
    results(n_idx).max_rand1 = max_rand1;
    results(n_idx).max_rand2 = max_rand2;
    results(n_idx).min_utility = min_utility;
    results(n_idx).out_sample_unif = out_sample_unif;
    results(n_idx).out_sample_eb = out_sample_eb;
    results(n_idx).out_sample_rand1 = out_sample_rand1;
    results(n_idx).out_sample_rand2 = out_sample_rand2;
    results(n_idx).out_sample_robust = out_sample_robust;
    results(n_idx).min_prior_type = min_prior_type; % Store the type of minimizing prior
end

% Set global default font to Times New Roman
% Set global default font and font size
set(0, 'DefaultAxesFontName', 'Times New Roman');  % Axis font
set(0, 'DefaultAxesFontSize', 14);                % Axis font size (including ticks)
set(0, 'DefaultTextFontName', 'Times New Roman');  % Text font (titles, labels, etc.)
set(0, 'DefaultTextFontSize', 14);                % Text font size
set(0, 'DefaultLegendFontName', 'Times New Roman'); % Legend font
set(0, 'DefaultLegendFontSize', 14);              % Legend font size


%% First Figure: Comprehensive Figure with Four Subplots
figure('Position', [100, 100, 1200, 900]);
% figure;
n_idx = 1;

% (a) log marginal likelihood
subplot(2, 2, 1)
hold on;
plot(results(n_idx).y_values, results(n_idx).loglik_unif, 'b-o', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).loglik_eb, 'r--s', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).loglik_rand1, 'g-d', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).loglik_rand2, 'm-^', 'LineWidth', 1.5);
% plot(results(n_idx).y_values, results(n_idx).loglik_eb + log(thre), 'k--*', 'LineWidth', 1.5);
hold off;
xlabel('Number of observed state $s_1$', 'Interpreter', 'latex');
xticks(results(n_idx).y_values);
ylabel('Log Marginal Likelihood');
title('(a) Log Marginal Likelihood','FontSize', 15);
grid on;
box on;
% legend({'Non-informative', 'EB', 'Informative1', 'Informative2', sprintf('log(\\alpha)+EB', thre)}, 'Location', 'best');
legend({'Non-informative', 'EB', 'Informative1', 'Informative2'}, 'Location', 'south','FontSize', 10);

% (b) worst-case prior
subplot(2, 2, 2)
hold on;
min_prior_types = results(n_idx).min_prior_type;
y_values = results(n_idx).y_values;

% Plot each prior type
for y_idx = 1:length(min_prior_types)
    prior_type = min_prior_types{y_idx};
    switch prior_type
        case 'Non-informative'
            scatter(y_values(y_idx), 1, 100, 'o', ...
                'MarkerFaceColor', 'b', ...
                'MarkerEdgeColor', 'k');
        case 'EB'
            scatter(y_values(y_idx), 2, 100, 's', ...
                'MarkerFaceColor', 'r', ...
                'MarkerEdgeColor', 'k');
        case 'Informative1'
            scatter(y_values(y_idx), 3, 100, 'd', ...
                'MarkerFaceColor', 'g', ...
                'MarkerEdgeColor', 'k');
        case 'Informative2'
            scatter(y_values(y_idx), 4, 100, '^', ...
                'MarkerFaceColor', 'm', ...
                'MarkerEdgeColor', 'k');
    end
end

hold off;
yticks(1:4);
yticklabels({'Non-informative', 'EB', 'Informative1', 'Informative2'});
ylim([0.5 4.5]);
xlabel('Number of observed state $s_1$', 'Interpreter', 'latex');
xticks(results(n_idx).y_values);
ylabel('Worst-case prior');
title('(b) Worst-case Prior','FontSize', 15);
grid on;
box on;

% (c) in-sample comparison
subplot(2, 2, 3)
hold on;
plot(results(n_idx).y_values, results(n_idx).max_unif, 'b-o', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).max_eb, 'r--s', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).max_rand1, 'g-d', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).max_rand2, 'm-^', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).min_utility, 'k--*', 'LineWidth', 1.5);
hold off;
xlabel('Number of observed state $s_1$', 'Interpreter', 'latex');
xticks(results(n_idx).y_values);
ylabel('In-sample value');
title('(c) In-sample comparison','FontSize', 15);
legend({'Non-informative', 'EB', 'Informative1', 'Informative2', 'Worst-case'}, 'Location', 'best','FontSize', 10);

grid on;
box on;

% (d) out-of-sample comparison
subplot(2, 2, 4)
hold on;
plot(results(n_idx).y_values, results(n_idx).out_sample_unif, 'b-o', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).out_sample_eb, 'r--s', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).out_sample_rand1, 'g-d', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).out_sample_rand2, 'm-^', 'LineWidth', 1.5);
plot(results(n_idx).y_values, results(n_idx).out_sample_robust, 'k--*', 'LineWidth', 1.5);
hold off;
xlabel('Number of observed state $s_1$', 'Interpreter', 'latex');
xticks(results(n_idx).y_values);
ylabel('Out-of-sample value');
ylim([min(value)-0.01,max(value)+0.01]);
title('(d) Out-of-sample comparison','FontSize', 15);
grid on;
box on;
% sgtitle(sprintf('Experiment %d: Beta(%d,%d)', experiment_idx, alpha_info_global1, beta_info_global1), ...
        % 'FontSize', 16, 'FontWeight', 'bold');

%%
% ===== Helper Functions =====
function [alpha_hat, beta_hat, log_likelihood] = empirical_bayes_beta_binomial(y, n)
    % Empirical Bayes estimation for Beta-Binomial model.
    % Estimates alpha and beta by maximizing the marginal likelihood.
    %
    % Inputs:
    %   y             - Observed successes (vector or scalar)
    %   n             - Number of trials (scalar or vector matching y)
    %   initial_guess - [alpha_init, beta_init] (optional, default [1, 1])
    %   options       - Optimization options (optional, from optimset)
    %
    % Outputs:
    %   alpha_hat     - Estimated alpha parameter
    %   beta_hat      - Estimated beta parameter
    %   log_likelihood - Maximized log marginal likelihood
    
    % Negative log-likelihood function for optimization (to minimize)
    neg_log_lik = @(params) -sum(beta_binomial_marginal_loglikelihood(y, n, params(1), params(2)));
    
    % Initial guess (uniform prior)
    initial_params = [1, 1];
     % Constrain alpha, beta > 0 using fmincon (requires Optimization Toolbox)
    lb = [1e-3, 1e-3];  % Lower bounds (alpha, beta >= 1e-3)
    ub = [Inf, Inf];     % No upper bounds
    
    % Optimize (requires Optimization Toolbox)
    options = optimset('Display', 'iter', 'MaxFunEvals', 1000);
    estimated_params = fmincon(neg_log_lik, initial_params, [], [], [], [], lb, ub, [], options);
    
    % Extract estimates
    alpha_hat = estimated_params(1);
    beta_hat = estimated_params(2);
    
    % Compute final log-likelihood
    log_likelihood = -neg_log_lik(estimated_params);
end



% marginal loglikelihood of Beta-binomial model
function log_p = beta_binomial_marginal_loglikelihood(y, n, alpha, beta)
    % Compute marginal likelihood for Beta-Binomial model (log-space for stability)
    log_p = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) + ...
            betaln(y + alpha, n - y + beta) - betaln(alpha, beta);
end