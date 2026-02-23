%
% 
%       Title: Ambiguity learning in sequential decision making with
%              parameter uncertainty
%              @Yue Shi, September 2022
%
%       Task 1: Investigate the impact of sample size on both robust and
%               out-of-sample values
%               a) plot robust and out-of-sample values vs. N for both
%               multiple-priors model and single-prior model
%       Issue: Without tuning, there is almost no difference btw our model
%              and the standard Bayesian model in out-of-sample
%              performance!
%              So, in this task, we only plot for our proposed model

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
[reward,Prob] = problemSet(numOfState,numOfAction,idxCase);


% Initial distribution
p0 = rand(1,numOfState);
p0 = p0/sum(p0);

% Obtain true value function and optimal policy
[optV_true, optA_true] = policyIter_gen(reward,Prob,numOfState,numOfAction,discount);
outVtrue = sum(p0.*optV_true);



%% Hyper-parameters
% Case 1: kexi = 1, 2norm(lambda0,ptrue)<= thre_i
% Case 2: kexi = 10, 2norm(lambda0,ptrue)<= thre_i

lambda0_MP = cell(1,numOfAction);
for a = 1:numOfAction
    lambda0_MP{a} = [zeros(numOfState,numOfState),ones(numOfState,numOfState)];
end
hypara = lambda0_MP;

% Common parameter
kexi0 = numOfState;         

% 60% dataset for training; 40% dataset for validation; use Ptrue for
% testing
ratio = 1;
task = 1; % given alpha value

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
% define output
numOfSampleSeq = [1 5 10 50 100 500 1000 5000]; % For N = 5000, it requires almost 2days for case 2; For N = 10000, requires very long computation time
output_MP = cell(1,length(numOfSampleSeq)+1);
outValue_MP = zeros(length(numOfSampleSeq),6+1); % mean value, 10%, 90%, (+1) average time
numOfInstance = 100;


for i_samp = 1:length(numOfSampleSeq)
    numOfSample = numOfSampleSeq(i_samp);
    alpha = 1/(numOfSample+1); 
    rhoN = kexi0/(kexi0+numOfSample*ratio);

    %%
    %%% Randomly generate 100 instance to compute average
    %%% performance
    
    value_multi = zeros(numOfInstance,2);
    time_multi = zeros(numOfInstance,1);

    t1 = clock;
    
    for i_inst = 1:numOfInstance
        t11 = clock;
        idxSeed = i_inst+numOfInstance;
        %%%%%%%%%%%%%%%%%%%%%%% Step 1: generate transition observations;
        rng(idxSeed);
        [nObs,nominalP,nominalP_valid,~] = generateObs_iid(numOfState,numOfAction,Prob,numOfSample,idxSeed,ratio);       
        
        %%%%%%%%%%%%%%%%%%%%%% Our approach: out-of-sample reward from
        %%%%%%%%%%%%%%%%%%%%%% multiple-priors model        
%         [value_multi(i_inst,:), time_multi(i_inst)] = frame_MP(numOfState,numOfAction,reward,Prob,discount,p0,...
%             nObs,nominalP_valid,alpha,kexi0,hypara,idxSet);
        
        [value_multi(i_inst,:),time_multi(i_inst)] = frame_MP_new(numOfState,numOfAction,reward,Prob,discount,p0,...
            nObs,nominalP,nominalP_valid,rhoN,alpha,kexi0,hypara,idxSet);
        
        
        t22 = clock;
        Etime = etime(t22,t11);
        
        %%%% output results
        formatSpec1 = 'Sample size is %d, instance is %d, time is %8.2f mins\n';
        fprintf(formatSpec1,numOfSample,i_inst,Etime/60);
    end
    
    outValue_MP(i_samp,:) = [mean(value_multi(:,1)),prctile(value_multi(:,1),[10,90]),...
        mean(value_multi(:,2)),prctile(value_multi(:,2),[10,90]),mean(time_multi)];
    
    disp(outValue_MP(:,1:6));
    
    output_MP{i_samp} = value_multi;
    output_MP{end} = outValue_MP;
    
    %%%% save data
    % save('data_task1_case2_MP.mat','output_MP');
    save('data_task1_case1_MP_new.mat','output_MP');
    
    t2 = clock;
    Etime = etime(t2,t1);
    formatSpec2 = 'Total time required for sample size %d is %8.2f hrs\n';
    fprintf(formatSpec2,numOfSample,Etime/3600);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
end



%% plot in-sample/out-of-sample value VS. sample size N (only MP model)
%%%%%%%%%%%%%%%%%%%%%%%%  Case 1
load('data_task1_case1_MP_new.mat','output_MP');
outValue_MP = output_MP{end};
 numOfSampleSeq = [1 5 10 50 100 500 1000 5000];
% in-sample
figure; 
tiledlayout(1,2,'TileSpacing','loose'); % loose; compact; tight; none
nexttile(1)
hold on;
x = 1:length(numOfSampleSeq); % equal space
% Trueth
p1 = plot(x,outVtrue+zeros(1,length(x)),'Color','k', 'LineStyle','--','linewidth',2);
% MP model
p2 = plot(x,outValue_MP(:,1),'Color','b', 'LineStyle','-','Marker','o','linewidth',1);
y1 = outValue_MP(:,2);
y2 = outValue_MP(:,3);
h = fill([x fliplr(x)],[y1' fliplr(y2')],'b');
set(h,'FaceColor','b','FaceAlpha',0.3,'EdgeColor','none');

legend([p1 p2],'$J_{{\bf P}^{true}}^*$','$\tilde{J}_N$',...
     'Interpreter', 'latex','location','southeast');

xticks(x)
set(gca,'XTickLabel',numOfSampleSeq); % x label
cap = xlabel({'Samples ($N$)','(a) $\tilde{J}_N$ versus $N$'});
set(cap,'Interpreter', 'latex');
ylim([6 9.2]);
yticks([6 7 8 9]);
ylabel('In-sample value');
box on;
hold off;
set(gca, 'fontsize', 18, 'fontname', 'Times New Roman');


% ouf-of-sample result
nexttile(2)
hold on;
x = 1:length(numOfSampleSeq); % equal space
% Trueth
p1 = plot(x,outVtrue+zeros(1,length(x)),'Color','k', 'LineStyle','--','linewidth',2);
% MP model
p2 = plot(x,outValue_MP(:,4),'Color','b', 'LineStyle','-','Marker','o','linewidth',1);
y1 = outValue_MP(:,5);
y2 = outValue_MP(:,6);
h = fill([x fliplr(x)],[y1' fliplr(y2')],'b');
set(h,'FaceColor','b','FaceAlpha',0.3,'EdgeColor','none');

legend([p1 p2],'$J^*_{{\bf P}^{true}}$','$J^{\tilde{\pi}^*_N}_{{\bf P}^{true}}$',...
     'Interpreter', 'latex','location','southeast');

xticks(x)
set(gca,'XTickLabel',numOfSampleSeq); % x label
xlabel('Samples ({\itN})');
cap = xlabel({'Samples ($N$)','(b) $J^{\tilde{\pi}^*_N}_{{\bf P}^{true}}$ versus $N$'});
set(cap,'Interpreter', 'latex');
ylim([6 9.2]);
yticks([6 7 8 9]);
ylabel('Out-of-sample value');
box on;
hold off;
set(gca, 'fontsize', 18, 'fontname', 'Times New Roman');




%%%%%%%%%%%%%%%%%%%%%%%%%  Case 2
load('data_task1_case2_MP.mat','outV_MP');
% numOfSampleSeq = [1 3 5 7 10 50 100 500 1000 5000];
outV_MP = outV_MP([1,3,5:end],:);
numOfSampleSeq = [1 5  10 50 100 500 1000 5000];
%color = get(gca,'colororder'); % 7 colors

% in-sample
figure; 
tiledlayout(1,2,'TileSpacing','loose'); % loose; compact; tight; none
nexttile(1)
hold on;
x = 1:length(numOfSampleSeq); % equal space
% Trueth
p1 = plot(x,outVtrue+zeros(1,length(x)),'Color','k', 'LineStyle','--','linewidth',2);
% MP model
p2 = plot(x,outV_MP(:,1),'Color','b', 'LineStyle','-','Marker','o','linewidth',1);
y1 = outV_MP(:,2);
y2 = outV_MP(:,3);
h = fill([x fliplr(x)],[y1' fliplr(y2')],'b');
set(h,'FaceColor','b','FaceAlpha',0.3,'EdgeColor','none');

% legend([p1 p2],'true','multiple-priors model',...
%     'location','southeast');

legend([p1 p2],'$J_{{\bf P}^{true}}^*$','$\tilde{J}_N$',...
     'Interpreter', 'latex','location','southeast');

xticks(x)
set(gca,'XTickLabel',numOfSampleSeq); % x label
cap = xlabel({'Samples ($N$)','(a) $\tilde{J}_N$ versus $N$'});
set(cap,'Interpreter', 'latex');
ylim([-60 -6]);
% yticks(-24:4:-8)
ylabel('In-sample value');
box on;
hold off;
set(gca, 'fontsize', 18, 'fontname', 'Times New Roman');


% ouf-of-sample result
nexttile(2)
hold on;
x = 1:length(numOfSampleSeq); % equal space
% Trueth
p1 = plot(x,outVtrue+zeros(1,length(x)),'Color','k', 'LineStyle','--','linewidth',2);
% MP model
p2 = plot(x,outV_MP(:,4),'Color','b', 'LineStyle','-','Marker','o','linewidth',1);
y1 = outV_MP(:,5);
y2 = outV_MP(:,6);
h = fill([x fliplr(x)],[y1' fliplr(y2')],'b');
set(h,'FaceColor','b','FaceAlpha',0.3,'EdgeColor','none');

% legend([p1 p2],'true','multiple-priors model',...
%     'location','southeast');

legend([p1 p2],'$J^*_{{\bf P}^{true}}$','$J^{\tilde{\pi}^*_N}_{{\bf P}^{true}}$',...
     'Interpreter', 'latex','location','southeast');

xticks(x)
set(gca,'XTickLabel',numOfSampleSeq); % x label
xlabel('Samples ({\itN})');
cap = xlabel({'Samples ($N$)','(b) $J^{\tilde{\pi}^*_N}_{{\bf P}^{true}}$ versus $N$'});
set(cap,'Interpreter', 'latex');
ylim([-60 -6]);
% yticks(-24:4:-8)
ylabel('Out-of-sample value');
box on;
hold off;
set(gca, 'fontsize', 18, 'fontname', 'Times New Roman');



%% potential code
%legend('$\mu_N(0.4|\textbf{\emph s}^N,\mu_0^1)$','$\mu_N(0.4|\textbf{\emph s}^N,\mu_0^2)$',...
%    'Interpreter', 'latex','location','northeast');




