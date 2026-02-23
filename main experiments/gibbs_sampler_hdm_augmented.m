function p_post_mean = gibbs_sampler_hdm_augmented(nObs,hyperprior)
    % X: G x K matrix of observed counts
    % alpha_init: initial value of alpha
    % gamma: Dirichlet prior concentration
    % num_iters: number of Gibbs iterations
    % Outputs:
    %   p_samples: num_iters x K matrix
    %   beta_samples: num_iters x K matrix
    %   alpha_samples: num_iters x 1 vector
rng(1);
    % Initialization
    X = nObs; 
    N = sum(X, 2);  % Fix: total counts per group (G x 1)
    num_iters = 5000;
    burn_in = 2000;
    
    [G, K] = size(X);  % G: num of group; K: count dimension % Default G =1
    beta = ones(1, K) / K;  % initialize base parameter 
    alpha = 1;     % initialize concentration parameter
    gamma = 1;    % initialize concentration parameter for Dirichlet hyperprior
    a0 = hyperprior(1);    % Gamma hyperprior for concentration
    b0 = hyperprior(2);    % Gamma hyperprior

    beta_samples = zeros(num_iters, K);
    alpha_samples = zeros(num_iters, 1);
    p_samples = zeros(num_iters, G, K);

    for iter = 1:num_iters
        % --- STEP 1: Sample CRT table counts t_{gk} ---
        T = zeros(G, K);  % table counts
        for g = 1:G
            for k = 1:K
                xgk = X(g, k);
                if xgk > 0
                    p = alpha * beta(k) ./ (alpha * beta(k) + (0:(xgk - 1)));
                    T(g, k) = sum(rand(1, xgk) < p);
                end
            end
        end

        % --- STEP 2: Sample beta from Dirichlet ---
        shape = gamma + sum(T, 1);  % posterior Dirichlet parameters
        beta = dirichlet_sample(shape);

         % --- Step 3: Sample alpha via data augmentation ---
        m = sum(T, 2);  % total table counts per group
        eta = betarnd(alpha, N);  % G x 1
        s = binornd(1, N ./ (N + alpha));  % G x 1
        shape_alpha = a0 + sum(m) - sum(s);
        rate_alpha = b0 - sum(log(eta));
        alpha = gamrnd(shape_alpha, 1 / rate_alpha);
        
        % --- STEP 4: Sample p from Dirichlet update ---
        p = zeros(G,K);
        for g = 1:G
            p(g,:) = dirichlet_sample(X(g,:) + alpha * beta);
        end
      
        
        % --- Save samples ---
        beta_samples(iter, :) = beta;
        alpha_samples(iter) = alpha;
        p_samples(iter,:,:) = p;
    end
%     
%     % Posterior mean plot to check sampling convergence
%     figure;
%     cumsum_vals = cumsum(p_samples(:,1,1));  % first component
%     running_mean = cumsum_vals ./ (1:length(cumsum_vals))';
%     plot(running_mean)
%     title('Running Mean of \theta_samples_1')
%     xlabel('Iteration')
%     ylabel('Mean')
    
    % Discard burn-in and return sample mean of p
    p_samples = p_samples(burn_in+1:end, :, :);
    p_post_mean = squeeze(mean(p_samples, 1));
    p_post_mean = p_post_mean';  % transfer to row vector
end


function lp = log_posterior_alpha(alpha, beta, X, a_alpha, b_alpha)
    [G, K] = size(X);
    lp = G * (gammaln(alpha) - gammaln(alpha + sum(X, 2))) + ...
         sum(sum(gammaln(alpha * beta + X) - gammaln(alpha * beta), 2)) + ...
         (a_alpha - 1) * log(alpha) - b_alpha * alpha;
end

function sample = dirichlet_sample(alpha_vec)
    y = gamrnd(alpha_vec, 1);
    sample = y / sum(y);
end
