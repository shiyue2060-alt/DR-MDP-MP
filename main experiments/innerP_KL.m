% function [obj_dual,p_worst] = innerP_KL(V,nominalP,radius,idx_dual_primal)
function [obj_dual,p_worst] = innerP_KL(V,nominalP,radius,prob)
% Option 1: solve dual problem but do not achieve optimality in extreme
% cases (e.g., r = [-200 -2 -10] 
% Option 2: solve primal problem directly


% %%%%Some issues regarding KL-divergence-based uncertainty set %%%%%%%%%%%%%%%%
% %%%%When only one observation is available, robust MDP with KL-based
% %%%%uncertainty set performs exactly the same as the nominal MDP. This is
% %%%%becasue KL-based uncertainty set does not allow scenarios pop out,
% %%%%i.e., \hat{p}_j=0 implies p_j = 0. So, in this case, it would be better
% %%%%to use other uncertainty sets, e.g., Burg entropy, Wasserstein
% %%%%distance.

% One observation!
% Note pj = 0 if \hat{p}_j=0 (can not pop out)
% So, we need to identify the index where \hat{p}_j = 0
p_worst = zeros(1,length(V));
idxnon0 = find(nominalP~=0);
V_in = V(idxnon0);
nominalP_in = nominalP(idxnon0);
if length(idxnon0) == 1 % we know the solution
    obj_dual = V(idxnon0);
    p_worst(idxnon0)=1;
    return;
elseif radius >= max(-log(nominalP_in)) % radius is too large and we just search the entire simplex (Nilim 2005 OR)
    [obj_dual,idx] = min(V_in);
    p_worst(idxnon0(idx(1))) = 1;
    return;
end

% if idx_dual_primal == 1 % Option 1: solve dual
if isempty(prob) == 1 % Option 1: solve dual
    %%
     [obj_dual,p_star_dual] = innerP_dual_KL(V_in, nominalP_in, radius);
     p_worst(idxnon0) = p_star_dual;
    
else % Option 2: solve primal by calling MOSEK
    %%
    %     t11 = clock;
    %     [obj_primal_mosek,p_star_mosek] = innerP_mosek_KL(V_in', nominalP_in', radius);
    %     obj_dual = obj_primal_mosek;
    %     p_worst(idxnon0) = p_star_mosek';
    n_states = length(V_in);
    prob.c = [V_in'; zeros(n_states, 1)];
    prob.buc(end) = radius;
    [r,res]=mosekopt('minimize echo(0)',prob);
    % --- 6. 检查解状态并提取结果 ---
    if strcmp(res.sol.itr.solsta, 'OPTIMAL')
        p_star_mosek = res.sol.itr.xx(1:n_states);
        %         fprintf('求解成功！KL散度: %.6f (约束: ≤ %.2f)\n', ...
        %                 sum(P_star .* log(P_star ./ P0)), theta);
    else
        fprintf('求解失败，解状态: %s\n', res.sol.itr.solsta);
        [obj_dual,p_star_dual] = innerP_dual_KL(V_in, nominalP_in, radius);
        p_star_mosek = p_star_dual';
        if sum(p_star_mosek.*log(p_star_mosek./nominalP_in))>radius
            error('MOSEK failed!\n');
        end
    end
    obj_primal_mosek = sum(p_star_mosek.*V_in');
    obj_dual = obj_primal_mosek;
    p_worst(idxnon0) = p_star_mosek';
    %     t22 = clock;
    % disp(etime(t22,t11));
end
end




