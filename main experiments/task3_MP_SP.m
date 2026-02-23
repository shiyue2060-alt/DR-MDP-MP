%
% 
%       Title: Ambiguity learning in sequential decision making with
%              parameter uncertainty
%              @Yue Shi, September 2022
%
%       Task 3: Compare the performance of multiple priors with the
%               single-prior model with different prior information
%               (1) kexi0 = 1,  D_KL(lambda0,ptrue)<= d_i, i = 1,2,3
%               (2) kexi0 = 10, D_KL(lambda0,ptrue)<= d_i, i = 1,2,3

%        Simulation example or machine repair example (Delage and Mannor 2010OR)!


clear
%clc
rng(1);
%%
% Input fixed data
idxCase = 2;
if idxCase == 1 % case 1: simulation
    numOfState = 30;      % number of states
    numOfAction = 10;     % number of actions
    discount = 0.9;
else % case 2: machine repair (Delage and Mannor2010OR)
    numOfState = 10;
    numOfAction = 2;
    discount = 0.8;
end

% Generate reward and transition probabilities
[reward,Prob,P_perturbed] = problemSet(numOfState,numOfAction,idxCase);


% Initial distribution
p0 = rand(1,numOfState);
p0 = p0/sum(p0);

% Obtain true value function and optimal policy
[optV_true, optA_true] = policyIter_gen(reward,Prob,numOfState,numOfAction,discount);
outVtrue = sum(p0.*optV_true);



%% Hyper-parameters
lambda0_MP = cell(1,numOfAction);
for a = 1:numOfAction
    lambda0_MP{a} = [zeros(numOfState,numOfState),ones(numOfState,numOfState)];
end

% generate single priors satisfying the prespecified distance from the
% trueth
thre2norm = [0.5 1 3]; % must NOT greater than 2; thre2norm = 3 means uniform prior mean

% lambda0_SP = cell(1,length(thre2norm));
% for x = 1: length(thre2norm)
%     lambda0_SP{x} = priorGenerate(numOfState,numOfAction,Prob,thre2norm(x));
% end
% % save('input_prior_c1','lambda0_SP'); % Case 1: keep it fixed for all experiments
% % save('input_prior_c2','lambda0_SP'); % Case 2: keep it fixed for all experiments

% Common parameter
kexiset = [numOfState/10, numOfState];  % fixed Dirichlet parameter
kexi0 = kexiset(1);    


% 60% dataset for training; 40% dataset for validation; use Ptrue for
% testing
ratio = 1;
task = 3; % plot out-of-sample values versus alpha

idxSet = [ratio, task]; % for task 2, length(idxSet) = 3;

%%%%%%%  Check the input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if task ~= 2 % task 1 and 3
    if length(idxSet)>2 || ratio < 1
        error('Error. ratio must be 1 and length(idxSet) must be 2 in tasks 1 and 3');
    end
else % task 2
    if scenario == 1 % scenario 1
        if ratio < 1
            error('Error. ratio must be 1 in scenario 1 in task 2');
        end
    else % scenarios 2 and 3
        if ratio == 1
            error('Error. ratio must be less than 1 in scenarios 2 and 3 in task 2');
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%
% Experiment setting
numOfSampleSeq = [3 6 9];
numOfInstance = 100;

%%% For a given sample size
alpha_set = [0.001 0.005 0.01 0.05 0.1 0.5 0.9];

%%
%%% Randomly generate 100 instance to compute average
%%% performance
t111 = clock;
%%

outValue_MP = zeros(length(alpha_set),3*length(numOfSampleSeq));
outValue_SP_hiera = zeros(length(thre2norm)+1,3*length(numOfSampleSeq)); % mean value, standard deviation,average time



if idxCase == 1 % simulation
    load('input_prior_c1','lambda0_SP');
else  % machine replacement problem   
    load('input_prior_c2','lambda0_SP'); 
end

for i_samp = 1:length(numOfSampleSeq)
    numOfSample = numOfSampleSeq(i_samp);
    rhoN = kexi0/(kexi0+numOfSample*ratio);
      
    time_MP = zeros(1,length(alpha_set)); % average computational time
    value_MP = zeros(numOfInstance,2*length(alpha_set)); %[inSample outSample]
    
    time_SP = zeros(1,length(thre2norm)+1);
    value_SP = zeros(numOfInstance,2*(length(thre2norm)+1));
    
    time_hiera_dir = 0;
    value_hiera_dir = zeros(numOfInstance,2);
    
    t1 = clock;
    for i_inst = 1:numOfInstance
        t11 = clock;
        idxSeed = i_inst+numOfInstance;
        %%%%%%%%%%%%%%%%%%%%%%% Step 1: generate transition observations;
        rng(idxSeed);
        [nObs,nominalP,nominalP_valid,~] = generateObs_iid(numOfState,numOfAction,Prob,numOfSample,idxSeed,ratio);
%         [nObs,nominalP,nominalP_valid,~] = generateObs_iid(numOfState,numOfAction,P_perturbed,numOfSample,idxSeed,ratio);
        
        %%%%%%%%%%%%%%%%%%%%%% Our approach: out-of-sample reward from
        %%%%%%%%%%%%%%%%%%%%% multiple-priors model             
        [value_MP(i_inst,:),oneTime_MP] = frame_MP_new(numOfState,numOfAction,reward,Prob,discount,p0,...
            nObs,nominalP,nominalP_valid,rhoN,alpha_set,kexi0,lambda0_MP,idxSet);
        time_MP = time_MP + oneTime_MP;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%  Benchmark 1: Single-prior model 
        [value_SP(i_inst,:),oneTime_SP] = frame_SP_new(numOfState,numOfAction,reward,Prob,discount,p0,...
            nObs,lambda0_SP,kexi0);
        time_SP = time_SP +oneTime_SP;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
        %%%%%%%%%%%%%%%%%%%%%  Benchmark 2: Hierarchical Bayesian model
        %%% 2: Dirichlet hyperprior and random concentration       
        hyperprior = [kexi0,1];
        [value_hiera_dir(i_inst,:),oneTime_hiera_dir] = frame_hierarchical(numOfState,numOfAction,reward,Prob,discount,p0,...
            nObs,hyperprior);
        time_hiera_dir = time_hiera_dir +oneTime_hiera_dir;
        
    
        t22= clock;
        Etime = etime(t22,t11);
        
        formatSpec1 = 'Sample size is %d, instance is %d, time is %8.2f mins\n';
        fprintf(formatSpec1,numOfSample,i_inst,Etime/60);
    end    
    
%     %%%%%%% Results for our model
    time_MP = time_MP/numOfInstance;     
  
    for x = 1: length(alpha_set)
        outValue_MP(x,3*(i_samp-1)+1:i_samp*3) = [mean(value_MP(:,2*x),1),std(value_MP(:,2*x),1),time_MP(x)];
    end
                            
                            
       %%%% Results for Single-prior model given different priors and hierarchical Bayes model  
   time_SP = time_SP/numOfInstance;
   time_hiera_dir = time_hiera_dir/numOfInstance;
   for x = 1: length(thre2norm)+1
       if x< length(thre2norm)+1  % SP given three different priors
           outValue_SP_hiera(x,3*(i_samp-1)+1:i_samp*3) = [mean(value_SP(:,2*x)),std(value_SP(:,2*x)),time_SP(x)];
       else  % Hierarchical Bayes model
           outValue_SP_hiera(x,3*(i_samp-1)+1:i_samp*3) = [ mean(value_hiera_dir(:,2)),std(value_hiera_dir(:,2)),time_hiera_dir];
       end
   end



t2 = clock;
Etime = etime(t2,t1);
formatSpec2 = 'Total time required for sample size %d is %8.2f hrs\n';
fprintf(formatSpec2,numOfSample,Etime/3600);
end
t222 = clock;
Etime = etime(t222,t111);




