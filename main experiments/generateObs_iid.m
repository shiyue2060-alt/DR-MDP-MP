function [nObs_train,nominalP_train,nominalP_valid,nObs]...
    = generateObs_iid(nState,nAction,Prob,numOfSample,idxSeed,ratio)
%% Generate realizations of the next state independently for each state-action pair given sample size
rng(idxSeed);
nObs = cell(1,nAction);
nObs_train = cell(1,nAction);
nominalP_train = cell(1,nAction);
nominalP_valid = cell(1,nAction);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Add perturbation on the observations by replacing Ptrue with P0 to
%%% generate data
%%

%%%%%%%%%%%%%%%%%%%% randomly generated MDP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%
% epsilon = 0.05;     % Noise level
% P_perturbed = Prob;
% for a = 1:nAction
%     noise = epsilon * randn(nState,nState);
%     tmp = Prob{a} + noise;
%     tmp = max(tmp,0);             % Enforce non-negativity and renormalize
%     P_perturbed{a} = tmp./sum(tmp,2);
% end
% Prob = P_perturbed;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%  machine repair problem %%%%%%%%%%%%%%%%%%%%%%%%
    %%

%     epsilon = 0.01;     % Noise level for comparison with single-prior model
% %     epsilon = 0.05;     % Noise level for comparison with RMDP
%     
%     P_perturbed = zeros(nState,nAction);
%     for s = 1:nState
%         if s<nState-2 % 1,...,7
%             P_perturbed(s,s) = 0.8;    % true = 0.2
%             P_perturbed(s,s+1) = 0.2;  % true = 0.8
%         elseif s==9
%             P_perturbed(s,s)=0.8;      % true = 0.2
%             P_perturbed(s,1)=0.2;      % true = 0.8
%         else
%             P_perturbed(s,s)=1;
%         end
%     end
%     
%     Prob_orig = P_perturbed;
%     noise = epsilon * randn(size(Prob_orig));
%     P_perturbed = Prob_orig + noise;
%     
%     % % Enforce non-negativity and renormalize
%     P_perturbed = max(P_perturbed,0);
%     P_perturbed = P_perturbed ./ sum(P_perturbed, 2);
%     
%     Prob{1} = P_perturbed;
%     % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%
t1 = clock;
for a = 1:nAction
    nObs_train_a = zeros(nState,nState);
    nObs_valid_a = zeros(nState,nState);
    Prob_a = Prob{a};
    Prob_a_cum = [zeros(size(Prob_a,1),1),cumsum(Prob_a,2)]; 
    for s = 1:nState
        if ratio < 1 % task 2
            for j = 1:numOfSample
                [~,~,s_next] = histcounts(rand,Prob_a_cum(s,:)); % histogram bin counts
                if j <= numOfSample*ratio
                    nObs_train_a(s,s_next) = nObs_train_a(s,s_next)+1;
                else
                    nObs_valid_a(s,s_next) = nObs_valid_a(s,s_next)+1;
                end
            end
            nObs{a} = nObs_train_a + nObs_valid_a;
            nominalP_valid{a} = nObs_valid_a./sum(nObs_valid_a,2);
        else %% task 1 needs very large sample size
            [nObs_train_a(s,:),~,~] = histcounts(rand(1,numOfSample),Prob_a_cum(s,:)); % histogram bin counts
        end
    end
    nObs_train{a} = nObs_train_a;
    nominalP_train{a} = nObs_train_a./sum(nObs_train_a,2);
end
t2 = clock;
Etime = etime(t2,t1);
end


