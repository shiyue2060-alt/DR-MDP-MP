function [lambdaOpt] = innerP_mosek_MP(prob,nState)
% Solve the robust counterpart of a given state-action part using MOSEK

% Case 1: lb <= lambda <= ub

% Omega = {1,...,S}, OmegaT = {s \in Omega: n(s) > 0}

% objectiveV = sum_s(lambda_s*V_s)

% minimize     sum_s(lambda_s*V_s)
% subject      sum_{s \in OmegaT} sum_k log(lambda_s*Lambda+k) >= rhs
%              sum_s lambda_s == 1
%              lb <= lambda_s <= ub, s in Omega


% Equivalent to
% minimize     sum_s(lambda_s*V_s)
% subject      (lambda_s*Lambda, 1, v(s,k))   in  Kexp, k = 0,...,n_s-1, s in OmegaT
%              sum_{s in OmegaT} sum_k v(s,k) = rhs
%              sum_s lambda_s == 1
%              lb <= lambda_s <= ub, s in Omega

% We order variables as: X = (lambda1,...,lambdaS, v(s,k)_{s \in OmegaT, k = 0,...,n_s})

t1 = clock;
% Optimize and print results
% [r,res]=mosekopt('minimize log(worst_prior.txt) echo(0)',prob);
%  [r,res]=mosekopt('minimize',prob);
[r,res]=mosekopt('minimize echo(0)',prob);

% %%%%%%%%% 2024-11-25 %%%%%%%%%%
% param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1.0e-3; % primal feasibility tolerance
% param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1.0e-3; % dual feasibility tolerance
% param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1.0e-3; % set relatie gap tolerance (double parameter)
% [r,res]=mosekopt('minimize echo(0)',prob,param);

% % Demonstrate information items after optimization
% [r,res] = mosekopt('minimize info', prob,param);
% 
% res.info.MSK_DINF_OPTIMIZER_TIME
% res.info.MSK_IINF_INTPNT_ITER

lambdaOpt = res.sol.itr.xx(1:nState)';
 
%  % Expected result: The solution status of the interior-point solution is optimal.
%  
%  % Check if we have non-error termination code or OK
%  if isempty(strfind(res.rcodestr, 'MSK_RES_ERR'))
%      
%      solsta = strcat('MSK_SOL_STA_', res.sol.itr.solsta);
%      
%      if strcmp(solsta, 'MSK_SOL_STA_OPTIMAL')
%          %fprintf('An optimal interior-point solution is located:\n');
%          lambdaOpt = res.sol.itr.xx(1:nState)';
%          
%          
%      elseif strcmp(solsta, 'MSK_SOL_STA_DUAL_INFEASIBLE_CER')
%         fprintf('Dual infeasibility certificate found.');
%          
%      elseif strcmp(solsta, 'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')
%         fprintf('Primal infeasibility certificate found.');
%          
%      elseif strcmp(solsta, 'MSK_SOL_STA_UNKNOWN')
%          The solutions status is unknown. The termination code
%         % indicates why the optimizer terminated prematurely.
%         fprintf('The solution status is unknown.\n');
%         fprintf('Termination code: %s (%d) %s.\n', res.rcodestr, res.rcode, res.rmsg);
%      elseif strcmp(solsta, 'MSK_SOL_STA_UNKNOWN') && (res.rcode == 100006)
%          %%% res.rcode == 100006: slow progress, often stalling happens near the optimum
%          idx = find(nObs>0);
%          if (length(idx) == 1) && (V(idx) == min(V(ub>0)))
%              lambdaOpt = zeros(1,nState);
%              lambdaOpt(idx) = 1;
%          else
%              lambdaOpt = res.sol.itr.xx(1:nState)';
%          end
%      else
%          fprintf('An unexpected solution status is obtained.');
%      end
%      
%  else
%      fprintf('Error during optimization: %s (%d) %s.\n', res.rcodestr, res.rcode, res.rmsg);
%  end


% objOpt = sum(lambdaOpt.*V);
% t2 = clock;
% Etime2 = etime(t2,t1);
% disp(Etime2);
% flag = 0;
% if Etime2 > 0.01
%     disp(Etime2);
%     flag = 1;
% end
end
