function [outV,outT] = frame_SP_new(numOfState,numOfAction,...
                    reward,Prob,discount,p0,nObs,lambda0_SP,kexi0)
           
% Compute posterior mean of the transition matrix
% !!! When lambda0 is obtained from empirical Bayes method, the posterior
% mean is exactly identical to the empirical frequency.
numPrior = length(lambda0_SP); % = 3 in task 3; 
outV = zeros(numPrior,2);
outT = zeros(1,numPrior);
for idx = 1:numPrior
    t1 = clock;
    lambda0 = lambda0_SP{idx};
    Prob_post = cell(1,numOfAction);
    for a = 1:numOfAction
        nObs_a = nObs{a};
        rho = kexi0./(kexi0+sum(nObs_a,2));
        q = nObs_a./sum(nObs_a,2);
        Prob_post_a = rho.*lambda0{a}+(1-rho).*q;
        Prob_post{a} = Prob_post_a;
    end 
    % Solve the MDP given Prob_post
    [inSampV,optAct,~] = ValueIter_gen(numOfState,numOfAction,...
        reward,discount,Prob_post);    
    % Evaluate the optimal policy under the true MDP
    outSampV = policyEva_gen(reward,Prob,optAct,numOfState,discount);
    outV(idx,:) = p0*[inSampV; outSampV]';
    t2 = clock;
    outT(idx) = etime(t2,t1);
end

%%% Add one for nominal policy
t11=clock;
Prob_nominal = cell(1,numOfAction);
for a = 1:numOfAction
    Prob_nominal{a} = nObs{a}./sum(nObs{a},2);
end
% Solve the MDP given Prob_post
[inSampV,optAct,~] = ValueIter_gen(numOfState,numOfAction,...
    reward,discount,Prob_nominal);
% Evaluate the optimal policy under the true MDP
outSampV = policyEva_gen(reward,Prob,optAct,numOfState,discount);
outV = [outV;p0*[inSampV; outSampV]'];
t22=clock;
outT = [outT,etime(t22,t11)];
outV = reshape(outV',[1,2*(numPrior+1)]);
end

