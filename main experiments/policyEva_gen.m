function valuefuc = policyEva_gen(reward,Prob,optAction,numOfState,discount)
% policy evaluation
tmpProb = zeros(numOfState,numOfState);
tmpReward = zeros(numOfState,1);
for s = 1: numOfState
    tmpProb(s,:) = Prob{optAction(s)}(s,:);  
    tmpReward(s) = reward(s,optAction(s));
end
A = eye(numOfState) - discount*tmpProb;
b = tmpReward;
valuefuc = transpose(A\b);
end

