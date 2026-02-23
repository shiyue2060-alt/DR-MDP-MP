function prob = prob_construct_mosek_rhs(kexi0,nObs,hypara)
%%% construct exponential conic program in MOSEK

% Equivalent to
% maximize     u
% subject      (lambda_s*Lambda, 1, v(s,k))   in  Kexp, k = 0,...,n_s-1, s in Omega*
%              sum_s sum_k v(s,k) = u
%              sum_s lambda_s == 1
%              lb <= lambda_s <= ub, s in Omega

% We order variables as: X = (lambda1,...,lambdaS, v(1,0),...,v(1,n1-1),
%                             ...,v(S,0),...,v(S,nS-1),u)


% Input data
nState = length(nObs);        % number of states
Nobs = sum(nObs);             % total number of transitions
nVar = nState+Nobs+1;         % total number of variables in CP


[r,res] = mosekopt('symbcon echo(0)');
% Objective
prob = [];
prob.c = [sparse(1, nVar-1),1]';

% Linear constraints:
prob.a = [sparse(1,nState),ones(1,Nobs),-1;
          ones(1,nState), sparse(1,Nobs+1)];
prob.blc = [0; 1];
% prob.buc = [0; 1];
prob.buc = [inf; 1];

% Case 1: lb <= lambda <= ub
lb = hypara(1:nState);
ub = hypara(nState+1:end);
% lb = hypara(1)+zeros(1,nState);
% ub = hypara(2)+zeros(1,nState);
prob.blx = [lb, -inf*ones(1,Nobs+1)];
prob.bux = [ub,inf*ones(1,Nobs+1)];


% Conic part FX + g in Kexp
% # exponential cones = Nobs
% (lambda_s*Lambda+k, 1, v(s,k))   in  Kexp, k = 0,...,n_s-1, s = 1,...,S
% (lambda1,...,lambdaS, v(1,0),...,v(1,n1-1),...,v(S,0),...,v(S,nS-1),u)
% [0 ... 0 Lambda 0    ...      0] + k
% [0              ...           0] + 1   in  Kexp
% [0     ...      0 ... 0 1 0...0] + 0




% %%%%%%%%%%%% Tooooo slow when N is large %%%%%%%%%%%%%%%%%%
% F = []; % must be a sparse matrix
% g = [];
% count = 0;
% for i = 1:nState
%     if nObs(i)>0
%         tmp = zeros(1,nState);
%         tmp(i) = kexi0;
%         line1 = [tmp,sparse(1,Nobs+1)];
%         line2 = sparse(1,nVar);
%         for j = 1:nObs(i)
%             count = count+1;
%             tmp = zeros(1,Nobs+1);
%             tmp(count) = 1;
%             line3 = [sparse(1,nState),tmp];
%             F = [F; line1;line2;line3];
%             g = [g; j-1;1;0];
%         end
%     end
% end

    %%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%
% F is a sparse matrix with (3N) by (S+N+1)
    rowidx = [3*(0:Nobs-1)+1,3*(1:Nobs)];
    colnidx = [repelem(1:nState,nObs),nState+(1:Nobs)];
    nonzeroidx = [kexi0+zeros(1,Nobs),ones(1,Nobs)];
    F = sparse(rowidx,colnidx,nonzeroidx,3*Nobs, nVar);
    
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

