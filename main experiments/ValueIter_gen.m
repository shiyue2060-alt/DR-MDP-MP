function [optValue,optAction,Qvalue] = ValueIter_gen(numOfState,numOfAction,...
    reward,discount,Prob)

% Machine Replacement Problem
% Value iteration algorithm given transition matrix (either true or
% nominal)

epsilon = 1e-2;
value1 = zeros(1,numOfState);   % value function at the nth iteration
value2 = value1;                % value function at the (n+1)th iteration
optAction = zeros(1,numOfState);% optimal action
Qvalue = zeros(numOfState,numOfAction);  % action value
iter = 1;

while true
   for a = 1:numOfAction
        Prob_a = Prob{a};
        Qvalue(:,a) = reward(:,a)+discount*Prob_a*value1';
    end
    [value2,optAction] = max(Qvalue');
    if (max(abs(value2 - value1))<epsilon) || (iter > 1000) % See Iyengar2005MOR for detailed derivation of precision
        optValue = value2;
        break;
    else
        iter = iter + 1;
        value1 = value2;
    end
end
end

