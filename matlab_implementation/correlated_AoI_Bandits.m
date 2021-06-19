% Code details:
% X- Latent Random Variable; prob_x - distribtion associated with latent
% random variable, G-bar - g_{i}(X) embedded in Matrix form.
% count1 x count2 x count3= Nuber of Iterations 
clearvars;
clc

% % Instance 1:
% 
% prob_x=[0.2 0.3 0.3 0.2];
% X=1:length(prob_x);
% G_bar=[1 0 0 0;...
%        0 1 1 0;...
%        0 0 1 1;...
%        0 0 0 1];
% [Number_of_Arms,~]=size(G_bar);
% mu_bar=G_bar*prob_x';
% 
% horizon_length=1e5;
% count1=5;
% count2=10;
% count3=10;

% Instance 2:

prob_x=[0.3 0.2 0.2 0.3];
X=1:length(prob_x);
G_bar=[1 0 0 0;...
       0 1 1 0;...
       0 0 1 1;...
       0 0 0 1];
[Number_of_Arms,~]=size(G_bar);
mu_bar=G_bar*prob_x';
horizon_length=1e5;
count1=5;
count2=10;
count3=10;


%% Instance 3:
% 
% prob_x=[0.1 0.1 0.6 0.2];
% X=1:length(prob_x);
% G_bar=[1 1 0 0;...
%          0 1 1 0;...
%          0 0 1 1];
% [Number_of_Arms,~]=size(G_bar);
% mu_bar=G_bar*prob_x';
% horizon_length=1e5;
% count1=5;
% count2=10;
% count3=10;

%% Instance 4:
% 
% prob_x=[0.05 0.05 0.6 0.3];
% X=1:length(prob_x);
% G_bar=[1 1 0 0;...
%          0 1 1 0;...
%          0 0 1 1];
% [Number_of_Arms,~]=size(G_bar);
% mu_bar=G_bar*prob_x';
% horizon_length=1e5;
% count1=5;
% count2=10;
% count3=10;

%% Instance 5:
% 
% prob_x=[0.05 0.8 0.05 0.1];
% X=1:length(prob_x);
% G_bar=[1 1 0 0;...
%          0 1 1 0;...
%          0 0 1 1];
% [Number_of_Arms,~]=size(G_bar);
% mu_bar=G_bar*prob_x';
% horizon_length=1e5;
% count1=5;
% count2=10;
% count3=10;

%% Instance 6:
% 
% prob_x=[0.1 0.1 0.5 0.3];
% X=1:length(prob_x);
% G_bar=[1 0 0 0;...
%        1 1 0 0;...
%        0 1 0 0;...
%        0 0 1 1];
% [Number_of_Arms,~]=size(G_bar);
% mu_bar=G_bar*prob_x';
% horizon_length=1e5;
% count1=5;
% count2=10;
% count3=10;

% %% Instance No Correlation
% 
% prob_x=[0.08 0.14 0.2 0.26 0.32];
% X=1:length(prob_x);
% G_bar=[1 0 0 0 0;...
%          0 1 0 0 0;...
%          0 0 1 0 0;...
%          0 0 0 1 0;...
%          0 0 0 0 1];
% [Number_of_Arms,~]=size(G_bar);
% mu_bar=G_bar*prob_x';
% horizon_length=1e5;
% count1=5;
% count2=10;
% count3=10;

%% UCB Algorithm
range_of_outputs=1;
UCB_AoI_1=zeros(count1,horizon_length+1);
for ll=1:count1
    UCB_AoI_2=zeros(count2,horizon_length+1);
    for jj=1:count2
        UCB_AoI_3=zeros(count3,horizon_length+1);
        for ii=1:count3
            UCB_indices=inf*ones(Number_of_Arms,1);
            T_k_t=zeros(Number_of_Arms,1); % Number of schdules vector for arms
            mu_hat=zeros(Number_of_Arms,1); % emperical reward associated with arms
            sample_X=datasample(X,horizon_length+1,'weights',prob_x); % generates sample of X with distribution prob_x
            v=zeros(1,horizon_length+1);
            v(1,1)=1;                               % Intial AoI Value
            for tt=2:horizon_length+1
        %         tt
                if tt<=Number_of_Arms+1
                    k_of_t=tt-1;
                    T_k_t(k_of_t,1)=T_k_t(k_of_t,1)+1;
                    reward=G_bar(k_of_t,sample_X(tt));
                    mu_hat(k_of_t,1)=reward;
                    v(1,tt)=((1+v(1,tt-1))*(1-reward))+reward;
                else
                    [~,k_of_t]=max(UCB_indices);
        %             k_of_t
                    T_k_t(k_of_t)=T_k_t(k_of_t,1)+1;
                    reward=G_bar(k_of_t,sample_X(tt));
                    mu_hat(k_of_t,1)=((mu_hat(k_of_t,1)*(T_k_t(k_of_t,1)-1))+reward)/T_k_t(k_of_t,1);
                    v(1,tt)=((1+v(1,tt-1))*(1-reward))+reward;
                end
        %         mu_hat
        %         T_k_t
                %% UCB Indices Update
                for kk=1:Number_of_Arms
                    if T_k_t(kk)==0
                        continue;
                    else
                        UCB_indices(kk,1)=mu_hat(kk,1)+(range_of_outputs*sqrt(2*log(tt)/T_k_t(kk)));
                    end 
                end
        %         UCB_indices
            end
            UCB_AoI_3(ii,:)=v;
        end
        UCB_AoI_2(jj,:)=mean(UCB_AoI_3,1);
    end
    UCB_AoI_1(ll,:)=mean(UCB_AoI_2);
end
Avg_UCB_AoI=mean(UCB_AoI_1,1);
% figure;stem(1:horizon_length+1,v);
% xlim([1 horizon_length+1])
clear UCB_reward2 C_UCB_reward3
%% Correlated UCB Algorithm
range_of_outputs=1;
display('C-UCB Started....');
C_UCB_AoI_1=zeros(count1,horizon_length+1);
for ll=1:count1
    C_UCB_AoI_2=zeros(count2,horizon_length+1);
    for jj=1:count2
        C_UCB_AoI_3=zeros(count3,horizon_length+1);
        for ii=1:count3
            UCB_indices=inf*ones(Number_of_Arms,1);
            T_k_t=zeros(Number_of_Arms,1); % Number of schdules vector for arms
            mu_hat=zeros(Number_of_Arms,1); % emperical reward associated with arms
            pseudo_rewards_hat=zeros(Number_of_Arms,Number_of_Arms); %% rows-> k_max columns-> Arms index
            sample_X=datasample(X,horizon_length+1,'weights',prob_x); % generates sample of X with distribution prob_x
            v=zeros(1,horizon_length+1);
            v(1,1)=1;                                   % Intial AoI Value
            for tt=2:horizon_length+1
        %         tt
                X_t=sample_X(tt);
                if tt<=Number_of_Arms+1
                    k_of_t=tt-1;
                    T_k_t(k_of_t,1)=T_k_t(k_of_t,1)+1;
                    reward=G_bar(k_of_t,sample_X(tt));
                    mu_hat(k_of_t,1)=reward;
                    v(1,tt)=((1+v(1,tt-1))*(1-reward))+reward;
                else
                    [~,indices]=max(T_k_t);
                    k_max=indices(randi(length(indices),1));
                    [~,Set_A]=find((pseudo_rewards_hat(k_max,:)-mu_hat(k_max))>=0);
        %             Set_A
                    UCB_indices_New=UCB_indices(Set_A);
                    [~,index]=max(UCB_indices_New);
                    k_of_t=Set_A(index(randi(length(index))));
                    T_k_t(k_of_t)=T_k_t(k_of_t,1)+1;
                    reward=G_bar(k_of_t,sample_X(tt));
                    mu_hat(k_of_t,1)=((mu_hat(k_of_t,1)*(T_k_t(k_of_t,1)-1))+reward)/T_k_t(k_of_t,1);
                    v(1,tt)=((1+v(1,tt-1))*(1-reward))+reward;
                end
        %         T_k_t
        %         mu_hat
                %% UCB Indices Update
                for kk=1:Number_of_Arms
                    if T_k_t(kk)==0
                        continue;
                    else
                        UCB_indices(kk,1)=mu_hat(kk,1)+(range_of_outputs*sqrt(2*log(tt)/T_k_t(kk)));
                    end 
                end
        %         UCB_indices
                %% Update pseudo rewards
                [~,x_set]=find(G_bar(k_of_t,:)==reward);
        %         x_set
                for kk=1:Number_of_Arms
                    if kk==k_of_t
                        pseudo_rewards_hat(k_of_t,kk)=mu_hat(kk);
                    else
                        s_k_k_of_t=max(G_bar(kk,x_set));
                        pseudo_rewards_hat(k_of_t,kk)=((pseudo_rewards_hat(k_of_t,kk)*(T_k_t(k_of_t)-1))+s_k_k_of_t)/T_k_t(k_of_t);
                    end
                end
        %         pseudo_rewards_hat
            end
            C_UCB_AoI_3(ii,:)=v;
        end
       C_UCB_AoI_2(jj,:)=mean(C_UCB_AoI_3,1); 
    end
    C_UCB_AoI_1(ll,:)=mean(C_UCB_AoI_2,1);
end
clear C_UCB_reward2 C_UCB_reward3
Avg_C_UCB_AoI=mean(C_UCB_AoI_1,1);
% figure;stem(1:horizon_length+1,v);
% xlim([1 horizon_length+1])
%% Optimal Reward Calculations
mu_max=max(mu_bar);
optimal_AoI=1/mu_max;
% Cummulative_optimal_reward2=max(G_bar*prob_x')*nn;
% figure;
% plot(Cummulative_optimal_reward1,'LineWidth',2);hold on;
% plot(Cummulative_optimal_reward2,'--r','LineWidth',2);
% legend({'1','2'},'FontSize',10);

%% Regret Calculations
Avg_Cummulative_UCB_regret=cumsum(Avg_UCB_AoI-optimal_AoI);
Avg_Cummulative_C_UCB_regret=cumsum(Avg_C_UCB_AoI-optimal_AoI);

%% Plot Results
h=figure;
plot(1:horizon_length+1,Avg_Cummulative_UCB_regret,'LineWidth',2); hold all;
plot(1:horizon_length+1,Avg_Cummulative_C_UCB_regret,'--','LineWidth',2);hold off;
xlabel('Horizon','FontSize',12);grid on;
ylabel('Regret','FontSize',12);xlim([1 horizon_length+1]);
legend({'UCB','C-UCB'},'FontSize',12);

%%
% h=figure;
% semilogy(Avg_C_UCB_reward,'LineWidth',2);hold all;
% semilogy(Avg_UCB_reward,'--','LineWidth',2);
% semilogy(optimal_reward,'.-','LineWidth',2);
% xlabel('Horizon','FontSize',12);grid on;
% ylabel('Cummulative Reward','FontSize',12);
% legend({'C-UCB','UCB','Optimal'},'FontSize',10);