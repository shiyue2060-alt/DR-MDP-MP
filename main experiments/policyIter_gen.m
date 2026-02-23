function [optValue, optAction] = policyIter_gen(reward,Prob,numOfState,numOfAction,discount)
%%%%% Policy iteration: compute optimal policy of standard MDP

Qvalue = zeros(numOfState,numOfAction);  % action value
v1 = zeros(1,numOfState);  % value function at the nth iteration
optA1 = ones(1,numOfState);% policy at the nth iteration
while true
    %%% policy evaluation
    v1 = policyEva_gen(reward,Prob,optA1,numOfState,discount);
    %%% policy improvement
    for a = 1:numOfAction
        Prob_a = Prob{a};
        Qvalue(:,a) = reward(:,a)+discount*Prob_a*v1';
    end
    [v2,optA2] = max(Qvalue');
    %%% stopping criteria
    if sum(optA2-optA1) == 0   % policy converges
        optAction = optA2;
        optValue = v2;
        break;
    else
        optA1 = optA2;
    end
end

end

