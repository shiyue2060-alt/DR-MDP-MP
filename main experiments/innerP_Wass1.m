function [obj_dual,p_star] = innerP_Wass1(V,nominalP,radius,prob)
% Option 1: solve primal problem directly
% Option 2: solve dual problem but do not achieve optimality in extreme
% cases (e.g., r = [-200 -2 -10]

% if idx_dual_primal == 1 % Option 1: solve dual
if isempty(prob) == 1 % Option 1: solve dual
    obj_dual = innerP_dual_Wass1(V, nominalP, radius);
    p_star = nominalP;
else % Option 2: solve primal
    %%%%%% solve LP directly and obtain obj_primal (too slow)
    % t1 = clock;
    % [obj_primal_lp,p_star_lp] = innerP_lp_Wass1(V', nominalP', radius);
    % obj_dual = obj_primal_lp;
    % t2 = clock;
    % disp(etime(t2,t1));
    
    
    %%%% call MOSEK to solve LP directly
%     t11 = clock;
%     [obj_primal_mosek,p_star_mosek] = innerP_mosek_Wass1(V', nominalP', radius);
%     obj_dual = obj_primal_mosek;
%     p_star = p_star_mosek';
%     % t22 = clock;
%     % disp(etime(t22,t11));
    n_states = length(V);
    prob.c = [V'; zeros(n_states^2, 1)];
    prob.buc(end) = radius;
    [r,res]=mosekopt('minimize echo(0)',prob);
    p_star = res.sol.bas.xx(1:n_states);
    obj_dual = sum(p_star.*V');

    
    % if abs(obj_primal_lp-obj_primal_mosek)>=1e-4 || max(abs(p_star_lp-p_star_mosek))>=1e-6
    %     xx = 1;
    % end
end
end



