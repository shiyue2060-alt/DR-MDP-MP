%
% 
%       Title: Ambiguity learning in sequential decision making with
%              parameter uncertainty
%              @Yue Shi, September 2022
%
%       Task 2: Compare the performance of multiple priors with
%               the robust MDP with some commonly used uncertainty set,
%               (1) KL-distance
%               (2) Wasserstein distance
%    Scenarios: (1) Use Ptrue to estimate out-of-sample value
%               (2) Separate the given sample into a training dataset and
%               validation dataset, and use the validation dataset to tune
%               hyparameter and use Ptrue to estimate out-of-sample value
%               (current practice. This one is the best!!!)
%               (3) Separate the given sample into a training dataset and
%               validation dataset, and use the validation dataset to tune
%               hyparameter and report the maximum one as the out-of-sample
%               value (Daniel Kuhn 2018 MP, called holdout method)

%        Simulation example or machine repair example (Delage and Mannor 2010OR)!


clear
%clc
rng(1);
%%
% Input fixed data
idxCase = 1;
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
% Case 1: lambda_lb <= lambda <= lambda_ub
% Without any priorir information, uniform prior is selected for
% single-prior model, and no restriction on the lambda0 set for
% multiple-priors model

lambda0_MP = cell(1,numOfAction);
for a = 1:numOfAction
    lambda0_MP{a} = [zeros(numOfState,numOfState),ones(numOfState,numOfState)];
end
hypara = lambda0_MP;


% Common parameter
kexi0 = numOfState/10;         


% alpha search range
a = 1:9;
b = -3:-1;
alpha = zeros(1,length(a)*length(b));
idx = 0;
for i = 1:length(b)
    for j = 1:length(a)
        idx = idx + 1;
        alpha(idx) = a(j)*10^b(i);
    end
end

% KL or Wass. radius search range
a = 1:10;
b = 0:5:9;
radiWass = [alpha, zeros(1,length(a)*length(b))];
for i = 1:length(a)
    for j = 1:length(b)
        idx = idx + 1;
        radiWass(idx) = a(i) + b(j)*0.1;
    end
end

% KL radius
radiKL = radiWass;

% 60% dataset for training; 40% dataset for validation; use Ptrue for
% testing
ratio = 0.6;
task = 2; % find the best alpha
scenario = 2;


idxSet = [ratio, task, scenario]; % for task 2, length(idxSet) = 3;

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
% define output
numOfSampleSeq = [5 10 15]; % Scenarios 2 & 3: *ratio (3/5) must be integer [3 6 9 12 15]

numOfInstance = 100;

output_MP = cell(1,length(numOfSampleSeq)+1);
output_Wass = cell(1,length(numOfSampleSeq)+1);
output_KL = cell(1,length(numOfSampleSeq)+1);

outValue_MP = zeros(length(numOfSampleSeq),3); % mean value, standard deviation,average time
outValue_Wass = zeros(length(numOfSampleSeq),3);
outValue_KL = zeros(length(numOfSampleSeq),3);



%%% 5 different sample size
for i_samp = 1:length(numOfSampleSeq)
    numOfSample = numOfSampleSeq(i_samp);
    rhoN = kexi0/(kexi0+numOfSample*ratio);
    %%
    %%% Randomly generate 100 instance to compute average
    %%% performance
    
    value_MP = zeros(numOfInstance,4);  % in-sample, out-sample, alpha,time
    value_Wass = zeros(numOfInstance,4);
    value_KL = zeros(numOfInstance,4);

    t1 = clock;
    
    for i_inst = 1:numOfInstance  % MP computation time increases using parfor
        idxSeed = i_inst+numOfInstance;
        t11 = clock;
        %%%%%%%%%%%%%%%%%%%%%%% Step 1: generate transition observations;
        rng(idxSeed);
        [nObs_train,nominalP_train,nominalP_valid,nObs] = ....
            generateObs_iid(numOfState,numOfAction,Prob,numOfSample,idxSeed,ratio);
%         [nObs_train,nominalP_train,nominalP_valid,nObs] = ....
%             generateObs_iid(numOfState,numOfAction,P_perturbed,numOfSample,idxSeed,ratio);
        
        %%%%%%%%%%%%%%%%%%%%%% Our approach: out-of-sample reward from
        %%%%%%%%%%%%%%%%%%%%%% multiple-priors model     
        [value_MP(i_inst,1:3),value_MP(i_inst,end)] = frame_MP_new(numOfState,numOfAction,reward,Prob,discount,p0,...
            nObs_train,nominalP_train,nominalP_valid,rhoN,alpha,kexi0,hypara,idxSet);
        
                
%         %%%%%%%%%%%%%%%%%%%% Benchmark 2: Robust MDP model with KL, Burg
%         %%%%%%%%%%%%%%%%%%%% entropy, or Wasserstein distance-based
%         %%%%%%%%%%%%%%%%%%%% uncertainty set
        idxOfSetType = 3; % 1: KL, 2: Burg entropy, 3: 1-dimentional Wasserstein
        [value_Wass(i_inst,1:3),value_Wass(i_inst,end)] = frame_Robust(numOfState,numOfAction,...
            reward,Prob,discount,p0,nominalP_train,nominalP_valid,radiWass,idxOfSetType,idxSet);
        
               
        idxOfSetType = 1; % 1: KL, 2: Burg entropy, 3: 1-dimentional Wasserstein
        [value_KL(i_inst,1:3),value_KL(i_inst,end)] = frame_Robust(numOfState,numOfAction,...
            reward,Prob,discount,p0,nominalP_train,nominalP_valid,radiKL,idxOfSetType,idxSet);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
       

        t22= clock;
        Etime = etime(t22,t11);
        formatSpec1 = 'Sample size is %d, instance is %d, time is %8.2f mins\n';
        fprintf(formatSpec1,numOfSample,i_inst,Etime/60);
        
    end
    
    outValue_MP(i_samp,:) = [mean(value_MP(:,2)),std(value_MP(:,2)),mean(value_MP(:,end))];
    
    outValue_Wass(i_samp,:) = [mean(value_Wass(:,2)),std(value_Wass(:,2)),mean(value_Wass(:,end))];
    
    outValue_KL(i_samp,:) = [mean(value_KL(:,2)),std(value_KL(:,2)),mean(value_KL(:,end))];
    
    
    output_MP{i_samp} = value_MP;
    output_MP{end} = outValue_MP;
     
    output_Wass{i_samp} = value_Wass;
    output_Wass{end} = outValue_Wass;
    
    output_KL{i_samp} = value_KL;
    output_KL{end} = outValue_KL;
    
    t2 = clock;
    Etime = etime(t2,t1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save data
%     save('data_task2_case1_MP_new.mat','output_MP');
%     save('data_task2_case1_Wass_new.mat','output_Wass');
%     save('data_task2_case1_KL_new.mat','output_KL');
    
end

%% plot
%color = get(gca,'colororder'); % 7 colors

%% ouf-of-sample result
% Our model Vs. Robust model, Only avg. and 10% percentile
figure; 
hold on;
x = 1:length(numOfSampleSeq); % equal space
% Trueth
p1 = plot(x,outVtrue+zeros(1,length(x)),'Color','k', 'LineStyle','--','linewidth',2);
% Our model
p2 = plot(x,outValue_MP(x,1+4),'Color','b', 'LineStyle','-','Marker','o','linewidth',1.5);
p2_1 = plot(x,outValue_MP(x,3+4),'Color','b', 'LineStyle','--','linewidth',1);

% Benchmark 2: robust MDP with Wasserstein distance
p3 = plot(x,outValue_Wass(:,1+4),'Color','m', 'LineStyle','-','Marker','o','linewidth',1.5);
p3_1 = plot(x,outValue_Wass(x,3+4),'Color','m', 'LineStyle','--','linewidth',1);

% Benchmark 2: robust MDP with KL distance
p4 = plot(x,outValue_KL(:,1+4),'Color','g', 'LineStyle','-','Marker','o','linewidth',1.5);
p4_1 = plot(x,outValue_KL(x,3+4),'Color','g', 'LineStyle','--','linewidth',1);


legend([p1 p2 p2_1 p3 p3_1 p4 p4_1],'true','multiple-priors model mean','multiple-priors model 10% percentile',...
    'RMDP with Wass. mean','RMDP with Wass. 10% percentile',...
    'RMDP with KL. mean','RMDP with KL. 10% percentile','location','southeast');


xticks(x)
set(gca,'XTickLabel',numOfSampleSeq); % x label
xlabel('Samples ({\itN})');
ylabel('Out-of-sample value');
box on;
hold off;
set(gca, 'fontsize', 14, 'fontname', 'Times New Roman');


% Our model Vs. Robust model, Only avg.
figure; 
hold on;
x = 1:length(numOfSampleSeq); % equal space
% Trueth
p1 = plot(x,outVtrue+zeros(1,length(x)),'Color','k', 'LineStyle','--','linewidth',2);
% Our model
p2 = plot(x,outValue_MP(x,1+4),'Color','b', 'LineStyle','-','Marker','o','linewidth',1.5);

% Benchmark 2: robust MDP with Wasserstein distance
p3 = plot(x,outValue_Wass(:,1+4),'Color','m', 'LineStyle','-','Marker','o','linewidth',1.5);

% Benchmark 2: robust MDP with KL distance
p4 = plot(x,outValue_KL(:,1+4),'Color','g', 'LineStyle','-','Marker','o','linewidth',1.5);


legend([p1 p2 p3 p4],'true','multiple-priors model',...
    'RMDP with Wass.','RMDP with KL.','location','southeast');


xticks(x)
set(gca,'XTickLabel',numOfSampleSeq); % x label
xlabel('Samples ({\itN})');
ylabel('Out-of-sample value');
box on;
hold off;
set(gca, 'fontsize', 14, 'fontname', 'Times New Roman');


%% potential code
%legend('$\mu_N(0.4|\textbf{\emph s}^N,\mu_0^1)$','$\mu_N(0.4|\textbf{\emph s}^N,\mu_0^2)$',...
%    'Interpreter', 'latex','location','northeast');









