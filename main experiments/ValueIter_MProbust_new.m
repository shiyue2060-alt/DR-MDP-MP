function [optValue,optAction,Qvalue]= ValueIter_MProbust_new(numOfState,numOfAction,...
    reward,discount,nominalP,RHS,rhoN,prob_mosek)
% Use robust value iteration to find optimal policy
% Need to solve the inner problem
value1 = zeros(1,numOfState);   % value function at the nth iteration
value2 = value1;                % value function at the (n+1)th iteration
optAction = zeros(1,numOfState);% optimal action
Qvalue = zeros(numOfState,numOfAction);  % action value

%% Robust value iteration algorithm

% Option 1: epsilon = 0.01 (very slow for large S, e.g., S=30)
% epsilon = 1e-2;
% t1 = clock;
% iter = 1;
% while true
%     for s = 1:numOfState
%         for a = 1:numOfAction
%             reward_sa = reward(s,a);
%             q_sa = nominalP{a}(s,:);
%             prob = prob_mosek{s,a};
%             prob.c(1:numOfState) = value1';
%             prob.blc(1) = RHS(s,a);
%             prob.buc(1) = RHS(s,a);
%             priorWorst_sa = innerP_mosek_MP(prob,numOfState);
%             tmp = (1-rhoN)*sum(q_sa.*value1)+ rhoN*sum(priorWorst_sa.*value1);
%             Qvalue(s,a) = reward_sa + discount*tmp;
%         end
%         [value2(s), optAction(s)] = max(Qvalue(s,:));
%     end
%     if (max(abs(value2 - value1))<epsilon) % See Iyengar2005MOR for detailed derivation of precision
%         optValue = value2;
%         break;
%     else
%         iter = iter+1;
%         value1 = value2;
%     end
% end
% 
% t2 = clock;
% timeE = etime(t2,t1);
% disp(timeE);



% % Option 2: epsilon = 0.1, save worst-case prior, then put them into standard VI to obtain in-sample value with improved precision
% epsilon = 1;  % for machine repair problem under task 2
% epsilon = 0.5;  % for machine repair problem under task 1
% epsilon = 0.1;  % for randomly generated MDP under task 2
epsilon = 0.5;    % for randomly generated MDP under task 1
t1 = clock;
iter = 1;
priorWorst = cell(1,numOfAction);
priorWorst_a = zeros(numOfState,numOfState);
while true       
    for a = 1:numOfAction     
        for s = 1:numOfState   % when S is large, use parfor; otherwise, use for
            reward_sa = reward(s,a);   
            q_sa = nominalP{a}(s,:);
            prob = prob_mosek{s,a};
            prob.c(1:numOfState) = value1';
            prob.blc(1) = RHS(s,a);
            prob.buc(1) = RHS(s,a);
            priorWorst_a(s,:) = innerP_mosek_MP(prob,numOfState);
            tmp = (1-rhoN)*sum(q_sa.*value1)+ rhoN*sum(priorWorst_a(s,:).*value1);
            Qvalue(s,a) = reward_sa + discount*tmp;
        end   
        priorWorst{a} = priorWorst_a;
    end
    for s = 1:numOfState
        [value2(s), optAction(s)] = max(Qvalue(s,:));
    end
    
%     disp(priorWorst{1});
%     disp(value2);
 
    if (max(abs(value2 - value1))<epsilon) % See Iyengar2005MOR for detailed derivation of precision
        posteriorWorst = priorWorst;
        for a = 1:numOfAction
            q_sa = nominalP{a}(s,:);
            posteriorWorst{a} = (1-rhoN).*q_sa+rhoN.*priorWorst{a};
        end
        [optValue,~,~] = ValueIter_gen(numOfState,numOfAction,reward,discount,posteriorWorst);
        break;
    else
        iter = iter+1;
        value1 = value2;
    end
end
% t2 = clock;
% timeE = etime(t2,t1);
% disp(timeE);
end

