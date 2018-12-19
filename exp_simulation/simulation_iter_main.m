iter_num = 20;
h_mle_vec = zeros(iter_num,1);
h_l2_vec = zeros(iter_num,1);
h_l1_vec = zeros(iter_num,1);
h_elastic_vec = zeros(iter_num,1);
h_theta_tilde_vec = zeros(iter_num,1);
h_theta_vec = zeros(iter_num,1);

for iteration=1:iter_num
    seed = iteration;
    run simulation.m
    h_mle_vec(iteration) = h_mle;
    h_l2_vec(iteration) = min(h_l2);
    h_l1_vec(iteration) = min(h_l1);
    h_elastic_vec(iteration) = min(h_elastic(:));
    h_theta_tilde_vec(iteration) = min(h_theta_tilde);
    h_theta_vec(iteration) = min(h_theta);
    iteration
    h_mle
    min(h_l2)
    min(h_l1)
    min(h_elastic(:))
    min(h_theta_tilde)
    min(h_theta)
end


mean(h_mle)
mean(h_l2_vec)
mean(h_l1_vec)
mean(h_elastic_vec)
mean(h_theta_tilde_vec)
mean(h_theta_vec)

corr_vec = [0.2,0.4,0.6,0.8];
h_mle_corr_vec = zeros(4,1);
h_l2_corr_vec = zeros(4,1);
h_l1_corr_vec = zeros(4,1);
h_elastic_corr_vec = zeros(4,1);
h_theta_tilde_corr_vec = zeros(4,1);
h_theta_corr_vec = zeros(4,1);

std_mle_corr_vec = zeros(4,1);
std_l2_corr_vec = zeros(4,1);
std_l1_corr_vec = zeros(4,1);
std_elastic_corr_vec = zeros(4,1);
std_theta_tilde_corr_vec = zeros(4,1);
std_theta_corr_vec = zeros(4,1);


nu_vec = [3,5,7,10];
for cc=1:4
    nu = nu_vec(cc);
    corr_tt = corr_vec(cc);
    iter_num = 20;
    h_mle_vec = zeros(iter_num,1);
    h_l2_vec = zeros(iter_num,1);
    h_l1_vec = zeros(iter_num,1);
    h_elastic_vec = zeros(iter_num,1);
    h_theta_tilde_vec = zeros(iter_num,1);
    h_theta_vec = zeros(iter_num,1);
    
    for iteration=1:iter_num
        seed = iteration;
        run simulation.m
        h_mle_vec(iteration) = h_mle;
        h_l2_vec(iteration) = min(h_l2);
        h_l1_vec(iteration) = min(h_l1);
        h_elastic_vec(iteration) = min(h_elastic(:));
        h_theta_tilde_vec(iteration) = min(h_theta_tilde);
        h_theta_vec(iteration) = min(h_theta);
        iteration
        h_mle
        min(h_l2)
        min(h_l1)
        min(h_elastic(:))
        min(h_theta_tilde)
        min(h_theta)
    end
    h_mle_corr_vec(cc) = mean(h_mle_vec);
    h_l2_corr_vec(cc) = mean(h_l2_vec);
    h_l1_corr_vec(cc) = mean(h_l1_vec);
    h_elastic_corr_vec(cc) = mean(h_elastic_vec);
    h_theta_tilde_corr_vec(cc) = mean(h_theta_tilde_vec);
    h_theta_corr_vec(cc) = mean(h_theta_vec);
    
    std_mle_corr_vec(cc) = std(h_mle_vec);
    std_l2_corr_vec(cc) = std(h_l2_vec);
    std_l1_corr_vec(cc) = std(h_l1_vec);
    std_elastic_corr_vec(cc) = std(h_elastic_vec);
    std_theta_tilde_corr_vec(cc) = std(h_theta_tilde_vec);
    std_theta_corr_vec(cc) = std(h_theta_vec);
end




corr_vec = [0.2,0.4,0.6,0.8];
h_mle_corr_vec = zeros(4,1);

std_mle_corr_vec = zeros(4,1);



nu_vec = [3,5,7,10];
for cc=1:4
    
    corr_tt = corr_vec(cc);
    iter_num = 20;
    h_mle_vec = zeros(iter_num,1);
    
    
    for iteration=1:iter_num
        seed = iteration;
        run simulation_mle.m
        h_mle_vec(iteration) = h_mle;
        
        
    end
    h_mle_corr_vec(cc) = mean(h_mle_vec);
   
    std_mle_corr_vec(cc) = std(h_mle_vec);
    
end

    




h_ratio = h_theta_vec ./ h_theta_tilde_vec;
[~,iteration] = min(h_ratio);
seed = iteration;
run simulation.m
h_mle_vec(iteration) = h_mle;
h_l2_vec(iteration) = min(h_l2);
h_l1_vec(iteration) = min(h_l1);
h_theta_tilde_vec(iteration) = min(h_theta_tilde);
h_theta_vec(iteration) = min(h_theta);
[~,ind] = min(h_theta);
t_vec = log(obj.t_seq);
[theta(:,ind),theta_tilde(:,ind)]

figure;
h1 = plot(t_vec,h_theta,'-*r','markersize',4,'linewidth',2);
hold on;
h2 = plot(t_vec,h_theta_tilde,':ob','markersize',4,'linewidth',2);
hold on;
h3 = plot(xlim,[1 1] * h_mle,':^k','linewidth',2);
xlim([min(t_vec),max(t_vec)]);
set(gca,'fontsize',20)
xlabel('$$\log{t}$$','Interpreter','latex','FontSize',30);
ylabel('$$\Vert \hat{\beta} - \beta \Vert_{2} / \Vert \beta \Vert_{2}$$',...
    'Interpreter','latex','FontSize',30);
title('Relative Error of $$\beta$$','Interpreter','latex','FontSize',35);
h = legend({'MSplit LBI ($$\beta$$)','MSplit LBI ($$\tilde{\beta}$$)','MLE'});
set(h,'Position',[0.6,0.5,0.1,0.3],'FontSize',30,'Interpreter','latex');


figure;
h1 = plot(t_vec,y_theta,'-*r','markersize',4);
hold on;
h2 = plot(t_vec,y_theta_tilde,':ob','markersize',4);
hold on;
h3 = plot(xlim,[1 1] * y_mle,':^g','linewidth',1.5);
xlim([min(t_vec),max(t_vec)]);
ylim([0,0.15]);
xlabel('$$\log{t}$$','Interpreter','latex','FontSize',20);
ylabel('$$\Vert X\hat{\beta} - X\beta \Vert_{2} / \Vert X\beta \Vert_{2}$$',...
    'Interpreter','latex','FontSize',20);
title('Relative Error of $$X\beta$$','Interpreter','latex');
h = legend({'Split LBI ($$\beta$$)','Split LBI ($$\tilde{\beta}$$)','MLE'});
set(h,'Position',[0.58,0.5,0.3,0.3],'FontSize',10,'Interpreter','latex');

    
    