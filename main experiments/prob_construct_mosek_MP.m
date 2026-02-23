function prob = prob_construct_mosek_MP(kexi0,nObs,hypara)
%%% construct exponential conic program in MOSEK

% Equivalent to
% minimize     sum_s(lambda_s*V_s)
% subject      (lambda_s*Lambda, 1, v(s,k))   in  Kexp, k = 0,...,n_s-1, s in OmegaT
%              sum_{s in OmegaT} sum_k v(s,k) = rhs
%              sum_s lambda_s == 1
%              lb <= lambda_s <= ub, s in Omega

% We order variables as: X = (lambda1,...,lambdaS, v(s,k)_{s \in OmegaT, k = 0,...,n_s})

% Input data
nState = length(nObs); % number of states
Nobs = sum(nObs);      % total number of state transitions
nVar = nState + Nobs;  % total number of variables
[r,res] = mosekopt('symbcon echo(0)');

lb = hypara(1:nState);
ub = hypara(nState+1:end);

%%% The following two values will be changed later in value iteration
%%% algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V = ones(1,nState);
rhs = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% lb = hypara(1)+zeros(1,nState);
% ub = hypara(2)+zeros(1,nState);

if Nobs == 0 % case 1: without reevaluation constraint (can be deleted to speed up)
    %%  LP
    % Objective
    prob = [];
    prob.c = V';
    
    % Linear constraints:
    prob.a = ones(1,nState);
    prob.blc = 1;
    prob.buc = 1;  

    prob.blx = lb;
    prob.bux = ub;
else % case 2: with reevaluation constraint
    %%  Conic P with exponential cone
    % Objective
    prob = [];
    prob.c = [V, sparse(1, Nobs)]';
    
    % Linear constraints:
    prob.a = [sparse(1,nState),ones(1,Nobs);
        ones(1,nState), sparse(1,Nobs)];
    prob.blc = [rhs; 1];
    prob.buc = [inf; 1];
    
    prob.blx = [lb, -inf*ones(1,Nobs)];
    prob.bux = [ub, inf*ones(1,Nobs)];
    
    
    % Conic part FX + g in Kexp
    % Nobs exponential cones
    % (lambda_s*Lambda+k, 1, v(s,k))   in  Kexp, k = 0,...,n_s-1, s in OmegaT
    
    % (lambda1,...,lambdaS, v(1,0),...,v(1,n1-1),...,v(S,0),...,v(S,nS-1))
    % [0 ... 0 Lambda 0    ...      0] + k
    % [0              ...           0] + 1   in  Kexp
    % [0     ...      0 ... 0 1 0...0] + 0
    
    
    %%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%
    % F is a sparse matrix with (3N) by (S+N)
    rowidx = [3*(0:Nobs-1)+1,3*(1:Nobs)];
    colnidx = [repelem(1:nState,nObs),nState+(1:Nobs)];
    nonzeroval = [kexi0+zeros(1,Nobs),ones(1,Nobs)];
    F = sparse(rowidx,colnidx,nonzeroval,3*Nobs, nVar);
    
    
    tmp = [];
    for i = 1:nState
        if nObs(i)>0
            tmp = [tmp,0:nObs(i)-1];
        end
    end
    g = reshape([tmp;ones(1,Nobs);zeros(1,Nobs)],Nobs*3,1);
    
    
    prob.f = F;
    prob.g = g;  
    prob.cones = repmat([res.symbcon.MSK_CT_PEXP, 3],1,Nobs);
end


end

