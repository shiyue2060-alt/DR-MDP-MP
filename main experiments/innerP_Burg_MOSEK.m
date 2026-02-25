function [objOpt,probW] = innerP_Burg_MOSEK(V,nominalP,radius)
% Solve the inner problem with Burg-based uncertainty set using MOSEK

% Burg entropy-based uncertainty set
% Omega = {1,...,S}
% minimize     sum_s(p_s*V_s)
% subject      (p_s, q_s, -x_s)   in  Kexp,  s in Omega
%              sum_{s in OmegaT} sum_k x_s = rhs
%              sum_s p_s == 1
%              p_s >= 0, s in Omega

% We order variables as: X = (p1,...,pS, x_1,...,x_S)


%t1 = clock;
% Input data
nState = length(nominalP); % number of states
nVar = 2*nState;  % total number of variables


[r,res] = mosekopt('symbcon');
% Objective
prob = [];
prob.c = [V, sparse(1, nState)]';

% Linear constraints:
prob.a = [sparse(1,nState),ones(1,nState);
          ones(1,nState), sparse(1,nState)];
prob.blc = [radius; 1];
prob.buc = [radius; 1];

prob.blx = [zeros(1,nState), -inf*ones(1,nState)];
prob.bux = inf*ones(1,nVar);


% Conic part FX + g in Kexp
% Nobs exponential cones
% (p_s, q_s, -x_s)   in  Kexp
% (lambda1,...,lambdaS, v(1,0),...,v(1,n1-1),...,v(S,0),...,v(S,nS-1))
% [0 ... 0 1 0        ...       0] + 0
% [0              ...           0] + q_s   in  Kexp
% [0     ...      0 ... 0 -1 0...0] + 0

F = [];
g = [];
zeroV = zeros(1,nState);       % in Line 1

for i = 1:nState
    zeroV_i = zeroV;
    zeroV_i(i) = 1;
    row1 = [zeroV_i, zeroV];
    row2 = sparse(1,nVar);
    row3 = [zeroV, -1*zeroV_i];
    F = [F; row1;row2;row3];
    g = [g; 0;nominalP(i);0];
end

prob.f = F;
prob.g = g;


prob.cones = repmat([res.symbcon.MSK_CT_PEXP, 3],1,nState);

% Optimize and print results
[r,res]=mosekopt('minimize echo(0)',prob);
% [r,res]=mosekopt('minimize',prob);
probW = res.sol.itr.xx(1:nState)';
objOpt = sum(probW.*V);
%t2 = clock;
%timeE = etime(t2,t1);
end

    

