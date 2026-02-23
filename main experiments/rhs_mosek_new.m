function objOpt = rhs_mosek_new(prob)
% Find the MLE of the Dirichlet-Multinomial parameters
% Use MOSEK to solve the resulting coinc exponential optimization 
%  
% Case 1: lb <= lambda <= ub

% Omega = {1,...,S}, Omega* = {s \in Omega: n(s) > 0}

% When there exist some states with zero transition, we ignore these states
% and only consider states in Omega*.
% We have lambdaOpt(s) = 0 if n(s) = 0.


% maximize     sum_s sum_k log(lambda_s*Lambda+k)
% subject      sum_s lambda_s == 1
%              lb <= lambda_s <= ub, s in Omega

% Equivalent to
% maximize     u
% subject      (lambda_s*Lambda, 1, v(s,k))   in  Kexp, k = 0,...,n_s-1, s in Omega*
%              sum_s sum_k v(s,k) = u
%              sum_s lambda_s == 1
%              lb <= lambda_s <= ub, s in Omega

% We order variables as: X = (lambda1,...,lambdaS, v(1,0),...,v(1,n1-1),
%                             ...,v(S,0),...,v(S,nS-1),u)


% Optimize and print results
% [r,res]=mosekopt('maximize',prob);
% [r,res]=mosekopt('maximize log(MLE_prior.txt) echo(0)',prob);
[r,res]=mosekopt('maximize echo(0)',prob);


%% set tolerance criteria
% param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1.0e-3; % primal feasibility tolerance
% param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1.0e-3; % dual feasibility tolerance
% param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1.0e-3; % set relatie gap tolerance (double parameter)
% [r,res]=mosekopt('maximize info',prob,param);
% 
% res.info.MSK_DINF_OPTIMIZER_TIME
% res.info.MSK_IINF_INTPNT_ITER

%% results
objOpt = res.sol.itr.pobjval; 
% lambdaOpt = res.sol.itr.xx(1:nState)';
end

