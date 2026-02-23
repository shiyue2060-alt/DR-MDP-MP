%%%%%%%% Illustrate Theorem 1: convergence of the all admissible posteriors  %%%%%%%%%%%%%%%
%%% Assumption 3: parameter space includes ptrue or not?
%%% Assumption 4: maximizer of the expected log-likelihood unique or not?

%%% Include the alpha value


%%%%%%%%%%% Required functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fun_posteri = @(theta,nObs,prior) (prod([theta, 1-theta].^nObs)*prior(1))...
    /(prod([theta, 1-theta].^nObs)*prior(1)+prod([1-theta, theta].^nObs)*prior(2));

fun_margli = @(theta,nObs,prior) (prod([theta, 1-theta].^nObs)*prior(1)+prod([1-theta, theta].^nObs)*prior(2));
alpha = 0.5;

%%
rng(2);
%%%%%%%%% Scenario 1: converge to the truth %%%%%%%%%%%%%%%%%%
prob_true = [0.4 0.6];
numOfObs = 100;

theta = 0.4; % 0.4 (seed 2)
prior = [0.5 0.5;0.8 0.2]; % parameter space {0.4,0.6}
numOfPriors = size(prior,1);

prob_cum = [0, cumsum(prob_true)];
nObs = zeros(1,length(prob_true));
output_posteri = zeros(numOfObs,2);

marglikeli = zeros(numOfPriors,1);

for i = 1:numOfObs
    [~,~,s_next] = histcounts(rand,prob_cum); % iid generated future state
    nObs(s_next) = nObs(s_next)+1;
    for j = 1:numOfPriors
        marglikeli(j) = fun_margli(theta,nObs,prior(j,:));
    end
    thre = alpha*max(marglikeli);
    idx = find(marglikeli>=thre);
    tmp = [];
    for x = 1: length(idx)
        tmp = [tmp, fun_posteri(theta,nObs,prior(idx(x),:))];
    end
    output_posteri(i,:) = [min(tmp), max(tmp)];
end
%subplot(1,3,1);
figure;
tiledlayout(1,3,'TileSpacing','compact'); % loose; compact; tight; none
nexttile(1)
hold on;
x = 1:numOfObs;
plot(x,output_posteri(:,2),'r--', 'linewidth',1.5);
plot(x,output_posteri(:,1),'b-', 'linewidth',1.5);
legend('$\mu_N^{max}(0.4)$','$\mu_N^{min}(0.4)$',...
    'Interpreter', 'latex','location','southeast');
% legend('$\mu_N(0.4|\textbf{\emph s}^N,\mu_0^1)$','$\mu_N(0.4|\textbf{\emph s}^N,\mu_0^2)$',...
%     'Interpreter', 'latex','location','east');
box on;
hold off;
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
ylim([0 1]);
xlabel('Samples ($N$)','Interpreter', 'latex');
title('(a) Scenario 1: $p^{true}=0.4$','Interpreter', 'latex');

%%
% clear;
rng(3);
%%%%%%%%% Scenario 2: not converge to the truth %%%%%%%%%%%%%%%%%%
prob_true = [0.55 0.45];
numOfObs = 100;

theta = 0.6; % 
prior = [0.5 0.5;0.8 0.2]; % parameter space {0.4,0.6}
numOfPriors = size(prior,1);

prob_cum = [0, cumsum(prob_true)];
nObs = zeros(1,length(prob_true));
output_posteri = zeros(numOfObs,2);
marglikeli = zeros(numOfPriors,1);


for i = 1:numOfObs
    [~,~,s_next] = histcounts(rand,prob_cum); % iid generated future state
    nObs(s_next) = nObs(s_next)+1;
    for j = 1:numOfPriors
        marglikeli(j) = fun_margli(theta,nObs,prior(j,:));
    end
    thre = alpha*max(marglikeli);
    tmp = [];
    for x = 1: length(idx)
        tmp = [tmp, fun_posteri(theta,nObs,prior(idx(x),:))];
    end
    output_posteri(i,:) = [min(tmp), max(tmp)];
end
%subplot(1,3,2);
nexttile(2)
hold on;
x = 1:numOfObs;
plot(x,output_posteri(:,2),'r--', 'linewidth',1.5);
plot(x,output_posteri(:,1),'b-', 'linewidth',1.5);
legend('$\mu_N^{max}(0.6)$','$\mu_N^{min}(0.6)$',...
    'Interpreter', 'latex','location','southeast');
box on;
hold off;
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
ylim([0 1]);
xlabel('Samples ($N$)','Interpreter', 'latex');
title('(b) Scenario 2: $p^{true}=0.55$','Interpreter', 'latex');

%%
rng(1);
%%%%%%%%% Scenario 3: no convergence %%%%%%%%%%%%%%%%%%
fun_posteri = @(theta,nObs,prior) (prod([theta, 1-theta].^nObs)*prior(1))...
    /(prod([theta, 1-theta].^nObs)*prior(1)+prod([1-theta, theta].^nObs)*prior(2));


prob_true = [0.5 0.5];
numOfObs = 100;

theta = 0.4; % 
prior = [0.5 0.5;0.8 0.2]; % parameter space {0.4,0.6}
numOfPriors = size(prior,1);

prob_cum = [0, cumsum(prob_true)];
nObs = zeros(1,length(prob_true));
output_posteri = zeros(numOfObs,2);
marglikeli = zeros(numOfPriors,1);


for i = 1:numOfObs
    [~,~,s_next] = histcounts(rand,prob_cum); % iid generated future state
    nObs(s_next) = nObs(s_next)+1;
    for j = 1:numOfPriors
        marglikeli(j) = fun_margli(theta,nObs,prior(j,:));
    end
    thre = alpha*max(marglikeli);
    tmp = [];
    for x = 1: length(idx)
        tmp = [tmp, fun_posteri(theta,nObs,prior(idx(x),:))];
    end
    output_posteri(i,:) = [min(tmp), max(tmp)];
end
%subplot(1,3,3);
nexttile(3)
hold on;
x = 1:numOfObs;
plot(x,output_posteri(:,2),'r--', 'linewidth',1.5);
plot(x,output_posteri(:,1),'b-', 'linewidth',1.5);
legend('$\mu_N^{max}(0.4)$','$\mu_N^{min}(0.4)$',...
    'Interpreter', 'latex','location','northeast');
box on;
hold off;
set(gca, 'fontsize', 20, 'fontname', 'Times New Roman');
ylim([0 1]);
xlabel('Samples ($N$)','Interpreter', 'latex');
title('(c) Scenario 3: $p^{true}=0.5$','Interpreter', 'latex');




