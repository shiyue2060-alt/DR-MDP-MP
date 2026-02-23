function [optValue,optAction,Qvalue]= ValueIter_robust(numOfState,numOfAction,...
    reward,discount,nominalP,radius,idxOfSetType,prob_mosek)
% Use robust value iteration to find optimal policy
% Need to solve the inner problem
value1 = zeros(1,numOfState);   % value function at the nth iteration
value2 = value1;                % value function at the (n+1)th iteration
optAction = zeros(1,numOfState);% optimal action
Qvalue = zeros(numOfState,numOfAction);  % action value

%% 

% Determine which type of uncertainty set is used
if idxOfSetType == 1 % KL
    f_innerP = @(Prob_sa,V,prob_mosek_sa) innerP_KL(V,Prob_sa,radius,prob_mosek_sa);
elseif idxOfSetType == 2 % Burg
    f_innerP = @(Prob_sa,V) innerP_Burg_MOSEK(V,Prob_sa,radius);
else % Wasserstein
    f_innerP = @(Prob_sa,V,prob_mosek_sa) innerP_Wass1(V,Prob_sa,radius,prob_mosek_sa);
end



%% Robust value iteration algorithm
% Option 1: epsilon = 0.01, solve dual
% Option 2: epsilon = 1, solve primal; terminate early to extract policy
% that has already converged!
if isempty(prob_mosek) == 1 % Option 1: solve dual
    %%
    epsilon = 1e-2;
    t1 = clock;
    iter = 1;
    while true
        for s = 1:numOfState
            for a = 1:numOfAction
                reward_sa = reward(s,a);
                Prob_sa = nominalP{a}(s,:);
                Qvalue(s,a) = reward_sa + discount*f_innerP(Prob_sa,value1,prob_mosek);
            end
            [value2(s), optAction(s)] = max(Qvalue(s,:));
        end
        
        if (max(abs(value2 - value1))<epsilon) % See Iyengar2005MOR for detailed derivation of precision
            optValue = value2;
            break;
        else
            iter = iter+1;
            value1 = value2;
        end
    end
    % t2 = clock;
    % timeE = etime(t2,t1);
    % disp(timeE);
    % disp(optValue);
    % disp(optAction);
    
else % Option 2: solve primal
    %%
%     epsilon = 0.5; % for machine repair problem
%     epsilon = 0.1; % for randomly generated MDP
    epsilon = 0.5; % for randomly generated MDP
    t1 = clock;
    iter = 1;
    
    P_worst = cell(1,numOfAction);
    P_worst_a = zeros(numOfState,numOfState);
    for a = 1:numOfAction
        P_worst{a} = P_worst_a;
    end
    
    while true
        for s = 1:numOfState
            for a = 1:numOfAction
                reward_sa = reward(s,a);
                Prob_sa = nominalP{a}(s,:);
                prob_mosek_sa = prob_mosek{s,a};
                [val, P_worst{a}(s,:)] = f_innerP(Prob_sa,value1,prob_mosek_sa);
                Qvalue(s,a) = reward_sa + discount*val;
            end
            [value2(s), optAction(s)] = max(Qvalue(s,:));
        end
    
        if (max(abs(value2 - value1))<epsilon) % See Iyengar2005MOR for detailed derivation of precision
            [optValue,~,~] = ValueIter_gen(numOfState,numOfAction,reward,discount,P_worst);
            break;
        else
            iter = iter+1;
            value1 = value2;
        end
    end
    t2 = clock;
    % timeE = etime(t2,t1);
    % disp(timeE);
    % disp(optValue);
    % disp(optAction);
end

end

