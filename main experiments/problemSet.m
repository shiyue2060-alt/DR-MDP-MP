function [reward,Prob,P_perturbed] = problemSet(numOfState,numOfAction,idxCase)
% Two problems: simulation and machine replacement (Delage and Mannor2010OR)
P_perturbed = [];
if idxCase ==1 % Case 1
%     rng(10);
rng;
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%% Garnet problem (Archibald et al., 1995)
%     reward = zeros(numOfState,numOfAction);
%     reward(:,2)=1;
%     reward(1,2)=0;
%     reward(1,1)=1;
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Generate reward for each state-action pair from U(0,1);
    reward = rand(numOfState,numOfAction);
    % Generate TRUE transition probabilities from U(0,1), then normalize
    Prob = cell(1,numOfAction);
    for a = 1:numOfAction
        tmp = rand(numOfState,numOfState);
        tmp = tmp./sum(tmp,2);
        Prob{a}=tmp;
    end
    
    
%     %%% Add perturbation on the observations by replacing Ptrue with P0 to generate data
%     k = 25;
%     P_perturbed = Prob;
%     for a = 1:numOfAction
%         [~,idx] = mink(reward(:,a),k);
%         P_perturbed{a}(:,idx) = 0;
%         P_perturbed{a} = P_perturbed{a}./sum(P_perturbed{a},2);
%     end
% 
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
else % Case 2: numOfState = 10, numOfAction=2
    reward = repmat([zeros(7,1);-200;-2;-10],1,2); % regardless of state
    Prob = cell(1,2);
    Prob_a1 = zeros(numOfState,numOfState);
    Prob_a2 = Prob_a1;
    
    for s = 1:numOfState
        if s<numOfState-2 % 1,...,7
            Prob_a1(s,s) = 0.2;
            Prob_a1(s,s+1) = 0.8;
            
            Prob_a2(s,s+1) = 0.3;
            Prob_a2(s,numOfState-1) = 0.6;
            Prob_a2(s,numOfState) = 0.1;
            
        elseif s==9
            Prob_a1(s,s)=0.2;
            Prob_a1(s,1)=0.8;
            
            Prob_a2(s,s) = 1;
           
        else
            Prob_a1(s,s)=1;
        end
        Prob_a2(8,:) = [zeros(1,7),0.3,0.6,0.1];
        Prob_a2(10,:) = [zeros(1,8),0.6,0.4];
        
    end


    Prob{1} = Prob_a1;
    Prob{2} = Prob_a2;
end
end

