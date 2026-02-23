function prob = prob_construct_mosek_Wass_KL(P0,idxOfSetType)
%%% construct primal problem in MOSEK for KL and Wasserstein
 % 输入:
    %   P0: 标称分布 (n_states x 1)
    %   theta: KL散度约束半径
    % 输出:
    %  prob: 锥优化问题构建
    
[r,res] = mosekopt('symbcon echo(0)');
theta = 1;

if idxOfSetType == 1 % KL
    %%
    P0 = P0(P0~=0);  % find the nonzero index
    n_states = length(P0);
    
    % --- 1. 定义MOSEK问题 ---
    prob = [];
    
    % 变量顺序: [P(1), ..., P(n), t(1), ..., t(n)]
    V = ones(n_states,1);
    prob.c = [V; zeros(n_states, 1)];  % 目标: min P' * V
    
    % --- 2. 定义约束矩阵 ---
    % (1) 概率归一化约束: sum(P) = 1
    A1 = [ones(1, n_states), zeros(1, n_states)];
    prob.a = A1;
    prob.blc = 1;  % 等式约束下界
    prob.buc = 1;  % 等式约束上界
    
    % (2) KL散度约束: sum(t) <= theta
    A2 = [zeros(1, n_states), ones(1, n_states)];
    prob.a = [prob.a; A2];
    prob.blc = [prob.blc; -inf];  % sum(t)无下界
    prob.buc = [prob.buc; theta]; % sum(t) <= theta
    
    % --- 3. 定义指数锥约束 ---
    % 每个约束形式: -t(s) >= P(s) * log(P(s)/P0(s))
    % 转化为指数锥: (P0(s), P(s), -t(s)) ∈ K_exp
    % Conic part FX + g in Kexp
    
    rowidx = [2+0:3:3*n_states,3:3:3*n_states];
    colnidx = 1:2*n_states;
    nonzeroval = [ones(1,n_states),-1*ones(1,n_states)];
    F = sparse(rowidx,colnidx,nonzeroval,3*n_states, n_states*2);
    prob.f = F;
    
    g = zeros(3*n_states,1);
    g(1:3:3*n_states) = P0;
    prob.g = g;
    
    prob.cones = repmat([res.symbcon.MSK_CT_PEXP, 3],1,n_states);
    
    % --- 4. 变量边界 ---
    prob.blx = [zeros(n_states, 1); -inf(n_states, 1)];  % P >= 0, t无下界
    prob.bux = inf*ones(2*n_states,1);  % 无上界
else  % == 3 Wass, solve LP
    %%
    %   d: 距离矩阵 (n_states x n_states)
    x_support = 1:length(P0);
    [X_i, X_j] = meshgrid(x_support, x_support);
    d = abs(X_i - X_j);  % cost matrix |x_i - x_j|
    
    n_states = length(P0);
    n_vars = n_states + n_states^2;  % 变量总数: P(s') + gamma(s', s'')
    
    % --- 1. 定义MOSEK问题 ---
    prob = [];
    
    % 目标函数: min sum_s' P(s') * V(s')
    V = ones(n_states,1);
    prob.c = [V; zeros(n_states^2, 1)];
    
    % 变量边界: P >= 0, gamma >= 0
    prob.blx = zeros(n_vars, 1);
    prob.bux = inf(n_vars, 1);
    
    % --- 2. 约束条件 ---
    % (1) sum_{s''} gamma(s', s'') = P(s')  (n_states个等式约束)
    A1 = zeros(n_states, n_vars);
    for s = 1:n_states
        A1(s, s) = -1;  % -P(s')
        for s2 = 1:n_states
            A1(s, n_states + (s-1)*n_states + s2) = 1;  % gamma(s', s'')
        end
    end
    prob.a = A1;
    prob.blc = zeros(n_states, 1);
    prob.buc = zeros(n_states, 1);
    
    % (2) sum_{s'} gamma(s', s'') = P0(s'')  (n_states个等式约束)
    A2 = zeros(n_states, n_vars);
    for s2 = 1:n_states
        for s = 1:n_states
            A2(s2, n_states + (s-1)*n_states + s2) = 1;  % gamma(s', s'')
        end
    end
    prob.a = [prob.a; A2];
    prob.blc = [prob.blc; P0];
    prob.buc = [prob.buc; P0];
    
    % (3) sum gamma(s', s'') * d(s', s'') <= theta  (1个不等式约束)
    A3 = zeros(1, n_vars);
    for s = 1:n_states
        for s2 = 1:n_states
            A3(1, n_states + (s-1)*n_states + s2) = d(s, s2);
        end
    end
    prob.a = [prob.a; A3];
    prob.blc = [prob.blc; -inf];
    prob.buc = [prob.buc; theta];
end



end

