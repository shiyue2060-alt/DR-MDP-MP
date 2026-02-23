function [outV,outT] = frame_Robust(numOfState,numOfAction,...
    reward,Prob,discount,p0,nominalP_train,nominalP_valid,radius,idxOfSetType,idxSet)
%%% When there exist multiple alpha values and compute the value function
%%% and policy for each alpha value
t1 = clock;
numOfRadius = length(radius);
inSampV = zeros(numOfRadius,numOfState);
outSampV = inSampV;
optA = inSampV;


%% 
% primal_dual index (default = 1)
% 1： solve inner use dual; fast but may lead to infeasible sol in extreme
% cases (e.g., r = [-200 -2 -10]
% 2: solve inner use primal: slow but optimal at least feasible is ensured
idx_dual_primal = 2; 

if idx_dual_primal == 1 % solve dual
    prob_mosek = [];
else  % solve primal
    prob_mosek = cell(numOfState,numOfAction);
    for a = 1:numOfAction
        for s = 1:numOfState
            prob_mosek{s,a} = prob_construct_mosek_Wass_KL(nominalP_train{a}(s,:)',idxOfSetType);
        end
    end  
end



func = @(radius_in) ValueIter_robust(numOfState,numOfAction,...
    reward,discount,nominalP_train,radius_in,idxOfSetType,prob_mosek);



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ratio = idxSet(1);
task = idxSet(2);
if length(idxSet)>2 % only for task = 2
    scenario = idxSet(3);
end

if ratio == 1 % use Ptrue to estimate out-of-sample value
    nominalP_valid = Prob;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if numOfRadius >1 % tasks 2 and 3
    for i = 1:numOfRadius  % use "for" for test
        radius_in = radius(i);
        [inSampV(i,:),optA(i,:),~] = func(radius_in);
        outSampV(i,:) = policyEva_gen(reward,nominalP_valid,optA(i,:),numOfState,discount);
    end
    
    if task > 2 % task 3: plot out-of-sample values versus alpha
        outV = [inSampV*p0',outSampV*p0'];
        outV = reshape(outV',1,[]);      
    else % task 2: tune hyperparameter alpha/radius using 3 approaches
        [~,idx] = max(outSampV*p0');
        % either use Ptrue or validation dataset to tune parameter and estimate out-of-sample value
        outV = [inSampV(idx,:)*p0', outSampV(idx,:)*p0',radius(idx)];
        if scenario == 2 % use validation dataset to tune parameter but use Ptrue to estimate out-of-sample value
            outV(2) = policyEva_gen(reward,Prob,optA(idx,:),numOfState,discount)*p0';
        end       
    end
    
else % task 1
    % Without parameter tuning, use TRUE transition matrix to evaluate in-sample policy
    radius_in = radius(1);
    [inSampV(1,:),optA(1,:),~]=func(radius_in);
    outSampV(1,:) = policyEva_gen(reward,Prob,optA(1,:),numOfState,discount);
    outV = p0*[inSampV; outSampV]';
end


t2 = clock;
outT = etime(t2,t1);
end



