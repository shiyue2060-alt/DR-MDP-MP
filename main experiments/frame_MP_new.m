function [outV,outT] = frame_MP_new(numOfState,numOfAction,reward,Prob,discount,p0,...
    nObs_train,nominalP_train,nominalP_valid,rhoN,alphaSeq,kexi0,hypara,idxSet)
%%% When there exist multiple alpha values and compute the value function
%%% and policy for each alpha value
t1 = clock;
numOfalpha = length(alphaSeq);
inSampV = zeros(numOfalpha,numOfState);
outSampV = inSampV;
optA = inSampV;

%%%%%%%%%%%% construct optimization problem used in MOSEK %%%%%%%%%%%%%%%%
%%%%%%%%%%%% ignore this time
prob_mosek = cell(numOfState,numOfAction);
probRHS_mosek = prob_mosek;

% t11 = clock;
for a = 1:numOfAction
    nObs_a = nObs_train{a};
    hypara_a = hypara{a};    
    for s = 1:numOfState      % when S is large, use parfor     
        probRHS_mosek{s,a} =  prob_construct_mosek_rhs(kexi0,nObs_a(s,:),hypara_a(s,:));
        prob_mosek{s,a} = prob_construct_mosek_MP(kexi0,nObs_a(s,:),hypara_a(s,:));
    end
    
end
% t12  =clock;
% Etime = etime(t12,t11);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: compute the RHS
RHS_new = cell(1,numOfalpha);
tmpRHS = zeros(numOfState,numOfAction);
for a = 1:numOfAction
    for s = 1:numOfState  % when S is large, use parfor
        tmpRHS(s,a) = rhs_mosek_new(probRHS_mosek{s,a});
    end
end
for i = 1:numOfalpha
    RHS_new{i} = tmpRHS+log(alphaSeq(i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ratio = idxSet(1);
task = idxSet(2);
if length(idxSet)>2 % only for task = 2
    scenario = idxSet(3);
end

if ratio == 1 % use Ptrue to estimate out-of-sample value
    nominalP_valid = Prob;
end


%%%%%%%%%%%%%%%%%%%%%%%% IISE revision %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if task == 2 % task 2: tune hyperparameter alpha using 3 approaches
    for i = 1:numOfalpha     
        RHS_in = RHS_new{i};
        [inSampV(i,:),optA(i,:),~]=ValueIter_MProbust_new(numOfState,numOfAction,...
            reward,discount,nominalP_train,RHS_in,rhoN,prob_mosek);
        outSampV(i,:) = policyEva_gen(reward,nominalP_valid,optA(i,:),numOfState,discount);
    end
    [~,idx] = max(outSampV*p0');
    % either use Ptrue or validation dataset to tune parameter and estimate out-of-sample value
    outV = [inSampV(idx,:)*p0', outSampV(idx,:)*p0',alphaSeq(idx)];
    if scenario == 2 % use validation dataset to tune parameter but use Ptrue to estimate out-of-sample value
        outV(2) = policyEva_gen(reward,Prob,optA(idx,:),numOfState,discount)*p0';
    end
    t2 = clock;
    outT = etime(t2,t1);
else % task 1 and 3, need to specify computation time for each alpha
    t3 = clock;
    outT = zeros(1,numOfalpha)+etime(t3,t1);
    for i = 1:numOfalpha     % in Task 2, use parfor only
        t111 = clock;
        RHS_in = RHS_new{i};
        [inSampV(i,:),optA(i,:),~]=ValueIter_MProbust_new(numOfState,numOfAction,...
            reward,discount,nominalP_train,RHS_in,rhoN,prob_mosek);
        outSampV(i,:) = policyEva_gen(reward,nominalP_valid,optA(i,:),numOfState,discount);
        t222 = clock;
        outT(i) = outT(i)+etime(t222,t111);
    end
    outV = [inSampV*p0',outSampV*p0']; 
    outV = reshape(outV',[1,2*numOfalpha]);
end

end



