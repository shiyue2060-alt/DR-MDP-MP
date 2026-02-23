function [outV,outT] = frame_hierarchical(numOfState,numOfAction,...
                    reward,Prob,discount,p0,nObs,hyperprior)
           
% Dirichlet-Multinomial model with Gamma hyperprior
% Use a Gibbs sampler with Metropolis-Hastings updats for statistical
% inference
% Use the posterior mean construct transition matrix
t1 = clock;

func = @(nObs_sa) gibbs_sampler_hdm_augmented(nObs_sa,hyperprior);


Prob_sample = cell(1,numOfAction);
Prob_sample_a = zeros(numOfState,numOfState);
for a = 1:numOfAction
    for s = 1:numOfState
        t11 = clock;
        Prob_sample_a(s,:) = func(nObs{a}(s,:));
        t12 = clock;
        outT2 = etime(t12,t11);
    end
    Prob_sample{a} = Prob_sample_a;
end

% t22 = clock;
% outT1 = etime(t22,t1);
% disp(outT1);

% Solve the MDP given Prob_sample
[inSampV,optAct,~] = ValueIter_gen(numOfState,numOfAction,reward,discount,Prob_sample);
% Evaluate the optimal policy under the true MDP
outSampV = policyEva_gen(reward,Prob,optAct,numOfState,discount);
outV = p0*[inSampV; outSampV]';
t2 = clock;
outT = etime(t2,t1);
% disp(outT);
end

