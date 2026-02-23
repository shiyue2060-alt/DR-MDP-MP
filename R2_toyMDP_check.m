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

% Select experiment indices to display
experiment_indices = [1, 2, 4, 5];

% Initialize cell array to store results for all experiments
all_results = cell(length(experiment_indices), 1);

% Perform calculations for each experiment index
for exp_idx_idx = 1:length(experiment_indices)
    experiment_idx = experiment_indices(exp_idx_idx);
    thre = thre_list(experiment_idx);

    % Parameters for the first informative prior
    alpha_info_global1 = k1(experiment_idx);
    beta_info_global1 = K - k1(experiment_idx);

    % Parameters for the second informative prior
    alpha_info_global2 = K - k1(experiment_idx);
    beta_info_global2 = k1(experiment_idx);

    % Initialize storage structure
    results = struct();

    for n_idx = 1:length(n_trials_list)
        n_data = n_trials_list(n_idx);
        y_values = 0:n_data;
        
        % Preallocate arrays
        loglik_unif = zeros(size(y_values));
        loglik_eb = zeros(size(y_values));
        loglik_rand1 = zeros(size(y_values));
        loglik_rand2 = zeros(size(y_values));
        utility_unif = zeros(size(y_values));
        utility_eb = zeros(size(y_values));
        utility_rand1 = zeros(size(y_values));
        utility_rand2 = zeros(size(y_values));
        max_unif = zeros(size(y_values));
        max_eb = zeros(size(y_values));
        max_rand1 = zeros(size(y_values));
        max_rand2 = zeros(size(y_values));
        min_utility = zeros(size(y_values));
        out_sample_unif = zeros(size(y_values));
        out_sample_eb = zeros(size(y_values));
        out_sample_rand1 = zeros(size(y_values));
        out_sample_rand2 = zeros(size(y_values));
        out_sample_robust = zeros(size(y_values));
        min_prior_type = cell(size(y_values));
        
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
            
            % --- Informative Prior 2 ---
            loglik_rand2(y_idx) = beta_binomial_marginal_loglikelihood(y_data, n_data, alpha_info_global2, beta_info_global2);
            p_predict = (y_data + alpha_info_global2) / (n_data + alpha_info_global2 + beta_info_global2);
            utility_rand2(y_idx) = sum(reward_a .* [p_predict, 1-p_predict]);
            
            % ===== Task Calculations =====
            % Task 1: Compare each prior's utility with the fixed utility
            % --- Uniform Prior ---
            if abs(utility_unif(y_idx) - fixed_utility) < 1e-6
                idx = action_opt;
                max_unif(y_idx) = max([utility_unif(y_idx), fixed_utility]);
            else
                [max_unif(y_idx), idx] = max([utility_unif(y_idx), fixed_utility]);
            end
            out_sample_unif(y_idx) = value(idx);

            % --- Empirical Bayes Prior ---
            if abs(utility_eb(y_idx) - fixed_utility) < 1e-6
                idx = action_opt;
                max_eb(y_idx) = max([utility_eb(y_idx), fixed_utility]);
            else
                [max_eb(y_idx), idx] = max([utility_eb(y_idx), fixed_utility]);
            end
            out_sample_eb(y_idx) = value(idx);

            % --- Random Prior 1 ---
            if abs(utility_rand1(y_idx) - fixed_utility) < 1e-6
                idx = action_opt;
                max_rand1(y_idx) = max([utility_rand1(y_idx), fixed_utility]);
            else
                [max_rand1(y_idx), idx] = max([utility_rand1(y_idx), fixed_utility]);
            end
            out_sample_rand1(y_idx) = value(idx);

            % --- Random Prior 2 ---
            if abs(utility_rand2(y_idx) - fixed_utility) < 1e-6
                idx = action_opt;
                max_rand2(y_idx) = max([utility_rand2(y_idx), fixed_utility]);
            else
                [max_rand2(y_idx), idx] = max([utility_rand2(y_idx), fixed_utility]);
            end
            out_sample_rand2(y_idx) = value(idx);
            
            % Task 2: Only compare priors with log_likelihood >= loglik_eb + log(thre)
            valid_prior_mask = false(1,4);
            loglik_threshold = loglik_eb(y_idx) + log(thre);
            
            if loglik_unif(y_idx) >= loglik_threshold
                valid_prior_mask(1) = true;
            end
            if ~isnan(loglik_eb(y_idx))
                valid_prior_mask(2) = true;
            end
            if loglik_rand1(y_idx) >= loglik_threshold
                valid_prior_mask(3) = true;
            end
            if loglik_rand2(y_idx) >= loglik_threshold
                valid_prior_mask(4) = true;
            end
            
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
            
            if isempty(valid_utilities)
                min_utility(y_idx) = fixed_utility;
                min_prior_type{y_idx} = 'None';
            else
                [min_val, min_idx] = min(valid_utilities);
                [min_utility(y_idx),idx] = max([min_val, fixed_utility]);
                out_sample_robust(y_idx) = value(idx);
                min_prior_type{y_idx} = prior_types{min_idx};
            end
        end
        
        % Store results
        results(n_idx).n_data = n_data;
        results(n_idx).y_values = y_values;
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
        results(n_idx).alpha_info_global1 = alpha_info_global1;
        results(n_idx).beta_info_global1 = beta_info_global1;
        results(n_idx).thre = thre;
    end
    
    all_results{exp_idx_idx} = results;
end

% Set global default fonts
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 14);
set(0, 'DefaultLegendFontName', 'Times New Roman');
set(0, 'DefaultLegendFontSize', 14);

%% Second Figure: Comprehensive Figure with Four Subplots - Out-of-sample Comparison
figure('Position', [100, 100, 1200, 900]);

for exp_idx_idx = 1:length(experiment_indices)
    experiment_idx = experiment_indices(exp_idx_idx);
    results = all_results{exp_idx_idx};
    n_idx = 1;
    
    subplot(2, 2, exp_idx_idx)
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
    ylim([min(value)-0.01, max(value)+0.01]);
    grid on;
    box on;
    
    % Modify subplot title format
    alpha_val = results(n_idx).alpha_info_global1;
    beta_val = results(n_idx).beta_info_global1;
    exponent = log10(results(n_idx).thre);  % Calculate
    
    subplot_letters = {'(a)', '(b)', '(c)', '(d)'};
    latex_str = sprintf('$({\\it{k}}_1^{Info},{\\it{k}}_2^{Info}) = (%d,%d)$', alpha_val, beta_val);
    % Combine
    title([subplot_letters{exp_idx_idx}, ' ', latex_str], 'Interpreter', 'latex', 'FontSize', 15);

    % Add legend only to the first subplot
    if exp_idx_idx == 1
        legend({'Non-informative', 'EB', 'Informative1', 'Informative2', 'Worst-case'}, ...
               'Location', 'west', 'FontSize', 10);
    end
end



%%
% ===== Helper Functions =====
function [alpha_hat, beta_hat, log_likelihood] = empirical_bayes_beta_binomial(y, n)
    neg_log_lik = @(params) -sum(beta_binomial_marginal_loglikelihood(y, n, params(1), params(2)));
    initial_params = [1, 1];
    lb = [1e-3, 1e-3];
    ub = [Inf, Inf];
    options = optimset('Display', 'off', 'MaxFunEvals', 1000);
    estimated_params = fmincon(neg_log_lik, initial_params, [], [], [], [], lb, ub, [], options);
    alpha_hat = estimated_params(1);
    beta_hat = estimated_params(2);
    log_likelihood = -neg_log_lik(estimated_params);
end

function log_p = beta_binomial_marginal_loglikelihood(y, n, alpha, beta)
    log_p = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) + ...
            betaln(y + alpha, n - y + beta) - betaln(alpha, beta);
end