%This code generates all the numbers and figures in Draft 2 of the paper.
%It is all in one place for ease of reference and to prevent old bugs and
%other issues from contaminating the results.


%First part of code runs the baseline model; second part of code runs the 
%model with insurance; and the third section performs the welfare analysis


%specifications:
%1) two types of housing and homeless state
%2) parameters not based on data
%3) states: 80 x 5 x 3 x 2 = 2400 (80 debt, 5 income, 3 housing states, 2 eviction states)
%4) no borrowing
%5) Rouwenhorst method for determining the transition probabilities


%housekeeping
clearvars
close all
clc

%set working directory
cd '/Users/claywagar/Box Sync/Research/Evictions/Theory Paper/Code/Draft 2/Full Paper Script/'

%for extra decimal places
format longg
%% PART I: baseline economy

%%%%%%%%%%%%%%%%%  Model Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha   = 0.55;    %consumption weight
sigma   = 2;       %coefficient of relative risk aversion
beta    = 0.95;    %subjective discount factor
r       = 0.01;    %quarterly interest rate
p1      = 50;      %price of cheap housing unit
p2      = 69;      %price of expensive housing unit
gamma   = 0.5;     %probability of being kicked out while in eviction proceedings
psi     = 0.44;    %probability of being redeemed
E       = 0.53;    %eviction costs
rho     = 0.97;    %income process persistence
sigma_y = 0.26;    %income process standard deviation



%%%%%%%%%%%%%%%%% Grid assignment %%%%%%%%%%%%%%%%
ND   = 80;    %debt grid points 
dmin = -4.5; 
dmax = 0;     %specified by collateral constraint
D    = linspace(dmin, dmax, ND); %debt grid
dval = [repmat(D,1)' repmat(D,1)' repmat(D,1)']; %for ease of reference in VFI

NH   = 3;       %number of housing options
h0   = 0.05;    %housing services of homeless state
h1   = 0.78;    %housing services of cheap apartment%
h2   = 0.92;    %housing services of expensive apaartment


hval = [repmat(h0,ND,1) repmat(h1,ND,1) repmat(h2,ND,1)]; %for ease of reference in VFI
H    = [h0 h1 h2];

NE   = 2;     %eviction states
%% Income Process, Transition Matrix and Initial Guesses

%number of income states
Ntotal = 5;

[y,Prob] = rouwenhorst(Ntotal,0,rho,sigma_y*sqrt(1-rho^2));
y        = exp(y);



%initial guesses for equilibrium objects    
vfunc    = zeros(ND,Ntotal,NH,NE);           %value function
oldvfunc = zeros(ND,Ntotal,NH,NE);
vN       = zeros(ND,Ntotal,NH,NE);           %value function for non default, only for non evicted state     
vD       = zeros(ND,Ntotal,NH,NE);           %value function for default, only for non evicted state
c        = ones(ND,Ntotal,NH,NE);            %consumption function
dp       = ones(ND,Ntotal,NH,NE);            %debt function
h        = ones(ND,Ntotal,NH,NE);            %housing function
cN       = ones(ND,Ntotal,NH,NE);            %consumption function 
dpN      = ones(ND,Ntotal,NH,NE);            %debt function
hN       = ones(ND,Ntotal,NH,NE);            %housing function
cD       = ones(ND,Ntotal,NH,NE);            %consumption function 
dpD      = ones(ND,Ntotal,NH,NE);            %debt function
hD       = ones(ND,Ntotal,NH,NE);            %housing function
DEF      = zeros(ND,Ntotal,NH,NE);          %default or not



%initial guess: uniform distribution
lambda     = zeros(ND,Ntotal,NH,NE)+1/(ND*Ntotal*NH*NE);

%default probability initial guess
pi       = [0.05 0.01];                  

    
%equilibrium rental prices initial guess
N1  =  pi(1) / (pi(1) + 2 * gamma);
R1 = p1 * r/((1-pi(1))*(1-N1) + gamma * N1) + (E * (1-N1) * pi(1))/((1-pi(1))*(1-N1) + gamma * N1);

N2 = pi(2) / (pi(2) + 2 * gamma);
R2 = p2 * r/((1-pi(2))*(1-N2) + gamma * N2) + (E * (1-N2) * pi(2))/((1-pi(2))*(1-N2) + gamma * N2);
%% Competitive equilibrium

%(1) solve for value and policy functions with intial guess for pi
%(2) derive stationary distribution by iterative procedure
%(3) calculate the implied unconditional default probability from this distribution
%(4) if the percent of households choosing to default is
%sufficiently close to pi then stop; if not, increase or decrease pi 
%by half the difference between pi and the implied default rate then return
%to (1)



%%%%%%%%%%%%%%%%%%%%  Technical parameters  %%%%%%%%%%%%%%%%%%%%%%%%%
tot_outfreq  = 1;         %Display frequency
tot_iter_tol = 250;       %Iteration tolerance
tot_tol      = 1e-5;      %Numerical tolerance
tot_iter     = 0;
tot_dev      = 100;
uptd         = 0.2;       %price updating parameter, keep low to prevent algorithm from fluctuating wildly


while tot_dev > tot_tol && tot_iter < tot_iter_tol
%% Value function iteration

    iter     = 0;
    tol      = 1e-6;
    iter_tol = 1000; 
    dev      = 100;
    outfreq  = 20;


    while dev > tol && iter < iter_tol


        for ii=1:ND
           for jj=1:Ntotal  
               for kk=1:NH
                  for ll=1:NE

               %case (i) good standing and housed, must choose whether or not
               %default

               if (kk == 2 || kk==3) && ll == 1
                   %consider no default case first

                   %consumption: unhoused state
                   c0 = (1/(1+r))*D + y(jj) - D(ii);
                   %consumption: cheap housing
                   c1 = (1/(1+r))*D + y(jj) - D(ii) - R1;
                   %consumption: expensive housing
                   c2 = (1/(1+r))*D + y(jj) - D(ii) - R2;
                   %concatenate
                   cboth = [c0' c1' c2'];

                   %keep consumption positive
                   cboth(cboth<0) = 0;

                   %utility, %stays in the same  house
                   u = ((cboth.^alpha .* hval.^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                   %bellman equation 
                   bellman0 = u(:,1) + beta * oldvfunc(:,:,1,1) * Prob(jj,:)' ; %continuation value recognizes choice of homeless state
                   bellman1 = u(:,2) + beta * oldvfunc(:,:,2,1) * Prob(jj,:)' ; %continuation value recognizes choice of cheap housed state
                   bellman2 = u(:,3) + beta * oldvfunc(:,:,3,1) * Prob(jj,:)' ; %continuation value recognizes choice of expensive housed state
                   bellNoDef     = [bellman0 bellman1 bellman2];
                   %pick level of consumption, housing and debt that maximizes the
                   %bellman equation and fill into the policy functions 
                   [argpol, arg]          = max(bellNoDef(:));
                   vN(ii,jj,kk,ll)        = argpol;
                   cN(ii,jj,kk,ll)        = cboth(arg);
                   dpN(ii,jj,kk,ll)       = dval(arg);
                   hN(ii,jj,kk,ll)        = hval(arg);

                   
                   %go on to default case

                   %consumption: housed state
                   c1 = (1/(1+r))*D + y(jj) ; %no rent paid and no assets

                   %keep consumption positive
                   c1(c1<0) = 0;

                   %utility
                   u = ((c1.^alpha .* H(kk).^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                   
                   %bellman equation
                   %tomorrow, with probability gamma you are evicted and
                   %homeless next period, with probability 1-gamma you stay in your house
                   bellDef = u' + beta * ((1-gamma) * oldvfunc(:,:,kk,2) * Prob(jj,:)' + gamma * oldvfunc(:,:,1,2) * Prob(jj,:)') ;

                   %pick level of consumption, housing and debt that maximizes the
                   %bellman equation and fill into the policy functions 
                   [argpol, arg]           = max(bellDef);
                   vD(ii,jj,kk,ll)         = argpol;
                   cD(ii,jj,kk,ll)         = c1(arg);
                   dpD(ii,jj,kk,ll)        = D(arg);
                   hD(ii,jj,kk,ll)         = H(kk);


                   %now fill in value and policy functions
                   if vN(ii,jj,kk,ll) > vD(ii,jj,kk,ll)
                       vfunc(ii,jj,kk,ll) = vN(ii,jj,kk,ll);
                       c(ii,jj,kk,ll)     = cN(ii,jj,kk,ll);
                       dp(ii,jj,kk,ll)    = dpN(ii,jj,kk,ll);
                       h(ii,jj,kk,ll)     = hN(ii,jj,kk,ll);
                       DEF(ii,jj,kk,ll)   = 0;  
                   else
                       vfunc(ii,jj,kk,ll) = vD(ii,jj,kk,ll);
                       c(ii,jj,kk,ll)     = cD(ii,jj,kk,ll);
                       dp(ii,jj,kk,ll)    = dpD(ii,jj,kk,ll);
                       h(ii,jj,kk,ll)     = hD(ii,jj,kk,ll);
                       DEF(ii,jj,kk,ll)   = 1;
                   end


               end



               %case (ii) good standing but unhoused, so kk = 1 and ll = 1,
               %essentially the same as the no default case above

               if kk == 1 && ll == 1

                   %consumption: homeless state
                   c0 = (1/(1+r))*D + y(jj) - D(ii);
                   %consumption: cheap housing state
                   c1 = (1/(1+r))*D + y(jj) - D(ii) - R1;
                   %consumption: expensive housing state
                   c2 = (1/(1+r))*D + y(jj) - D(ii) - R2;
                   %concatenate
                   cboth = [c0' c1' c2'];

                   %keep consumption positive
                   cboth(cboth<0) = 0;

                   %utility, %stays in the same  house
                   u = ((cboth.^alpha .* hval.^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                   %bellman equation 
                   bellman0 = u(:,1) + beta * oldvfunc(:,:,1,1) * Prob(jj,:)' ; %continuation value recognizes choice of homeless state
                   bellman1 = u(:,2) + beta * oldvfunc(:,:,2,1) * Prob(jj,:)' ; %continuation value recognizes choice of cheap housing state
                   bellman2 = u(:,3) + beta * oldvfunc(:,:,3,1) * Prob(jj,:)' ; %continuation value recognizes choice of expensive housing state
                   bell     = [bellman0 bellman1 bellman2];
                   %pick level of consumption, housing and debt that maximizes the
                   %bellman equation and fill into the policy functions 
                   [argpol, arg]             = max(bell(:));
                   vfunc(ii,jj,kk,ll)        = argpol;
                   c(ii,jj,kk,ll)            = cboth(arg);
                   dp(ii,jj,kk,ll)           = dval(arg);
                   h(ii,jj,kk,ll)            = hval(arg);
               end

               %case (iii) household defaulted last period or has been in
               %eviction but is housed, so kk = 2 and ll = 2

               if (kk == 2 || kk == 3) && ll == 2
                   %pays no rent but can accumulate assets
                  
                   %consumption: housed
                   c1 = (1/(1+r))*D + y(jj) - D(ii) ; %no rent but can accumulate assets
                   
                   %keep consumption positive
                   c1(c1<0) = 0;

                   %utility
                   u = ((c1.^alpha .* H(kk).^(1-alpha)).^(1-sigma)-1)./(1-sigma);
                    
                   %bellman equation 
                   bellman1 = u' + beta * ((1-gamma) * oldvfunc(:,:,kk,2) * Prob(jj,:)' + gamma * oldvfunc(:,:,1,2) * Prob(jj,:)') ;

                   %pick level of consumption, housing and debt that maximizes the
                   %bellman equation and fill into the policy functions 
                   [argpol, arg]           = max(bellman1);
                   vfunc(ii,jj,kk,ll)      = argpol;
                   c(ii,jj,kk,ll)          = c1(arg);
                   dp(ii,jj,kk,ll)         = D(arg);
                   h(ii,jj,kk,ll)          = H(kk);

               end

               %case (iv) household has been successfully evicted and is
               %unhoused, waiting to be redeemed to good standing

               if kk == 1 && ll == 2

                   %consumption: housed state
                   c1 = (1/(1+r))*D + y(jj) - D(ii) ; %no rent but can accumulate assets

                   %keep consumption positive
                   c1(c1<0) = 0;

                   %utility
                   u = ((c1.^alpha .* h0.^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                   %bellman equation 
                   bellman1 = u' + beta * ((1-psi) * oldvfunc(:,:,1,2) * Prob(jj,:)' + psi * oldvfunc(:,:,1,1) * Prob(jj,:)') ;

                   %pick level of consumption, housing and debt that maximizes the
                   %bellman equation and fill into the policy functions 
                   [argpol, arg]           = max(bellman1);
                   vfunc(ii,jj,kk,ll)      = argpol;
                   c(ii,jj,kk,ll)          = c1(arg);
                   dp(ii,jj,kk,ll)         = D(arg);
                   h(ii,jj,kk,ll)          = h0;

               end


                 end
              end
           end
        end


        %calculate distance
        dev = max(max(max(max(abs(vfunc - oldvfunc)))));
        iter = iter + 1;    

        %simple updating
        %oldvfunc = vfunc;

        %mcqueen porteus bounds and update,
        bunder = (beta/(1 - beta)) * min(min(min(min(vfunc - oldvfunc))));
        bover  = (beta/(1 - beta)) * max(max(max(max(vfunc - oldvfunc))));
        oldvfunc = vfunc + (bunder + bover)/2;




%         if mod(iter, outfreq) == 0
%         fprintf('%d          %1.7f \n',iter,dev);
%         fprintf('dp=  %1.8f  \n', min(min(min(min(dp)))))
%         fprintf('dp=  %1.8f  \n', max(max(max(max(dp)))))
%         fprintf('c=  %1.8f  \n', min(min(min(min(c)))))
%         fprintf('c=  %1.8f  \n', max(max(max(max(c)))))
%         end



    end


    fprintf('policy functions computed\n\n');
%% Time-invariant measure

    iter_lambda     = 0;
    iter_tol_lambda = 1000;
    outfreq_lambda  = 25;
    dev_lambda      = 100;
    tol_lambda      = 1e-6;


    while dev_lambda > tol_lambda && iter_lambda < iter_tol_lambda

       old_lambda = lambda;

       for i=1:ND
           for j=1:Ntotal
               for k=1:NH
                   for l=1:NE

                       value = 0;

                       for ii=1:ND
                           for jj=1:Ntotal
                              for kk=1:NH
                                  for ll=1:NE 

                                     %case (i) if you are housed in bad
                                     %standing you cannot transition to good
                                     %standing in this model and if you are
                                     %unhoused in bad standing you must wait to
                                     %be redeemed

                                      if (k == 2 || k == 3) && l == 1

                                          if (kk == 2 || kk == 3) && ll == 1

                                                value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i))* (h(ii,jj,kk,ll)==H(k)) * (1-DEF(ii,jj,kk));

                                          elseif kk==1 && ll==1

                                                value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k));

                                          end
                                      end
                                      
                                      
                                      

                                      %case (ii) 

                                      if k == 1 && l == 1

                                          if kk==1 && ll==1                                        
                                                value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k));   
                                                
                                          elseif kk==1 && ll==2
                                              %with probability psi the agent
                                              %transitions to good standing
                                                value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * psi * (dp(ii,jj,kk,ll)==D(i));
                                                
                                          elseif (kk==2 || kk==3) && ll==1
                                                value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k)); 
                                          end
                                      end

                                      %case (iii) if you are unhoused in
                                      %good standing you cannot transition to
                                      %this state, nor if you have been evicted
                                      %and are unhoused

                                      if (k == 2 || k == 3) && l == 2

                                          if (kk==2 || kk==3) && ll==2
                                                %remain in house with
                                                %probability 1-gamma
                                                value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (1-gamma) * (h(ii,jj,kk,ll)==H(k)) * (dp(ii,jj,kk,ll)==D(i));

                                          elseif (kk==2 || kk==3) && ll==1
                                                %defaulting agent moves to
                                                %housed but bad standing with
                                                %probability 1-gamma
                                                value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (1-gamma) * (dp(ii,jj,kk,ll)==D(i))* (h(ii,jj,kk,ll)==H(k)) * DEF(ii,jj,kk,ll);
                                          end
                                      end



                                      %case (iv) sum only over kk=2,ll=2, kk=1, ll=2 
                                      %and kk=2,ll=1 because agent cannot go from
                                      %unhoused in good standing to unhoused in
                                      %bad standing                                  

                                      if k == 1 && l == 2

                                          if (kk==2 || kk==3) && ll==2
                                              %agent is successfully evicted
                                              %with probablity gamma
                                              value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * gamma * (dp(ii,jj,kk,ll)==D(i)); 

                                          elseif kk==1 && ll==2
                                              %agent remains unhoused in bad
                                              %standing with probability 1-psi
                                              value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (1-psi) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k));

                                          elseif (kk==2 || kk==3) && ll==1
                                              %defaulting agent successfully
                                              %evicted with probability gamma
                                              value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * gamma * (dp(ii,jj,kk,ll)==D(i)) * DEF(ii,jj,kk,ll);                               
                                          end

                                      end



                                  end
                              end      
                           end
                        end


        lambda(i,j,k,l) = value;


                   end
               end
           end
       end

        %calculate distance
        dev_lambda = max(max(max(max(abs(lambda - old_lambda)))));
        iter_lambda = iter_lambda + 1; 

        if mod(iter_lambda, outfreq_lambda) == 0
        fprintf('%d          %1.7f \n',iter_lambda,dev_lambda);
        end


    end


    fprintf('\nstationary distribution computed\n\n');
%% Update

    implied_def_prob_total = sum(sum(sum(sum(DEF.*lambda))))/(sum(sum(lambda(:,:,2,1)))+sum(sum(lambda(:,:,3,1))));
    
    implied_def_prob_1 = sum(sum(DEF(:,:,2,1).*lambda(:,:,2,1)))/sum(sum(lambda(:,:,2,1)));
    implied_def_prob_2 = sum(sum(DEF(:,:,3,1).*lambda(:,:,3,1)))/sum(sum(lambda(:,:,3,1)));
    
    %calculate distance
    tot_dev = max([(abs(pi(1) - implied_def_prob_1)) (abs(pi(2) - implied_def_prob_2))]);
    tot_iter = tot_iter + 1; 
    
    %update default rate and prices
    pi(1) = pi(1) + uptd*(implied_def_prob_1 - pi(1));
    pi(2) = pi(2) + uptd*(implied_def_prob_2 - pi(2));
    
    %share of nonpaying
    N1  =  pi(1) / (pi(1) + 2*gamma);
    N2  =  pi(2) / (pi(2) + 2*gamma);
    
    %equilibrium rental price
    R1 = p1 * r/((1-pi(1))*(1-N1) + gamma*N1) + (E * (1-N1) * pi(1))/((1-pi(1))*(1-N1) + gamma*N1);
    R2 = p2 * r/((1-pi(2))*(1-N2) + gamma*N2) + (E * (1-N2) * pi(2))/((1-pi(2))*(1-N2) + gamma*N2);
    
    %average debt holding
    avedebt = sum(sum(sum(sum(dp.*lambda))));
    
    %percentage of time in expensive unit, paying
    time2 = sum(sum(lambda(:,:,3,1)));
    
    
    riskFreeRent1 = p1*r;
    riskFreeRent2 = p2*r;

    %percent above risk free rent
    riskpremium1 = (R1 - riskFreeRent1)/riskFreeRent1; 
    riskpremium2 = (R2 - riskFreeRent2)/riskFreeRent2; 
    
    if mod(tot_iter, tot_outfreq) == 0
        fprintf('%d          %1.7f \n',tot_iter,tot_dev);
        fprintf('implied def prob 1  %1.8f  \n', implied_def_prob_1)
        fprintf('implied def prob 2 %1.8f  \n', implied_def_prob_2)
        fprintf('pi(1)  %1.8f  \n', pi(1))
        fprintf('pi(2)  %1.8f  \n', pi(2))
        fprintf('R1  %1.8f  \n', R1)
        fprintf('R2  %1.8f  \n\n', R2)
        fprintf('Risk premium 1 %1.8f \n', riskpremium1)
        fprintf('Risk premium 2 %1.8f \n', riskpremium2)
        fprintf('Time in 2  %1.8f  \n', time2)
        fprintf('Average debt  %1.8f  \n', avedebt)
        fprintf('Total default rate %1.8f \n\n', implied_def_prob_total)
    end


end
%% Save output

save('solution_baseline')
%% PART II: economy with rental firm insurance

%housekeeping
clearvars
close all
clc


%%%%%%%%%%%%%%%%%  Model Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha   = 0.55;    %consumption weight
sigma   = 2;       %coefficient of relative risk aversion
beta    = 0.95;    %subjective discount factor
r       = 0.01;    %quarterly interest rate
p1      = 50;      %price of cheap housing unit
p2      = 69;      %price of expensive housing unit
gamma   = 0.5;     %probability of being kicked out while in eviction proceedings
psi     = 0.44;    %probability of being redeemed
E       = 0.53;    %eviction costs
rho     = 0.97;    %income process persistence
sigma_y = 0.26;    %income process standard deviation



%%%%%%%%%%%%%%%%% Grid assignment %%%%%%%%%%%%%%%%
ND   = 80;    %debt grid points 
dmin = -4.5; 
dmax = 0;     %specified by collateral constraint
D    = linspace(dmin, dmax, ND); %debt grid
dval = [repmat(D,1)' repmat(D,1)' repmat(D,1)']; %for ease of reference in VFI

NH   = 3;       %number of housing options
h0   = 0.05;    %housing services of homeless state
h1   = 0.78;    %housing services of cheap apartment%
h2   = 0.92;    %housing services of expensive apaartment


hval = [repmat(h0,ND,1) repmat(h1,ND,1) repmat(h2,ND,1)]; %for ease of reference in VFI
H    = [h0 h1 h2];

NE   = 2;     %eviction states
EV   = [1 2]; %eviction states: in good standing, in bad standing
%% Income Process, Transition Matrix and Initial Guesses

%number of income states
Ntotal = 5;

[y,Prob] = rouwenhorst(Ntotal,0,rho,sigma_y*sqrt(1-rho^2));
y        = exp(y);



%initial guesses for equilibrium objects    
vfunc    = zeros(ND,Ntotal,NH,NE);           %value function
oldvfunc = zeros(ND,Ntotal,NH,NE);
vN       = zeros(ND,Ntotal,NH,NE);           %value function for non default, only for non evicted state
oldvN    = zeros(ND,Ntotal,NH,NE);     
vD       = zeros(ND,Ntotal,NH,NE);           %value function for default, only for non evicted state
oldvD    = zeros(ND,Ntotal,NH,NE);
c        = ones(ND,Ntotal,NH,NE);            %consumption function
dp       = ones(ND,Ntotal,NH,NE);            %debt function
h        = ones(ND,Ntotal,NH,NE);            %housing function
cN       = ones(ND,Ntotal,NH,NE);            %consumption function 
dpN      = ones(ND,Ntotal,NH,NE);            %debt function
hN       = ones(ND,Ntotal,NH,NE);            %housing function
cD       = ones(ND,Ntotal,NH,NE);            %consumption function 
dpD      = ones(ND,Ntotal,NH,NE);            %debt function
hD       = ones(ND,Ntotal,NH,NE);            %housing function
DEF      = zeros(ND,Ntotal,NH,NE);          %default or not



%initial guess: uniform distribution
lambda     = zeros(ND,Ntotal,NH,NE)+1/(ND*Ntotal*NH*NE);

%insurance levy initial guess
tau = .01;                

%equilibrium rental prices, initial guess
R1 = (p1*r)/(1-tau);

R2 = (p2*r)/(1-tau);
%% Competitive equilibrium

%Outer loop
%(1) solve model (inner loop) with current guess of tau
%(2) update tau based on distance between tau and implied tau to balance 
%insurance payments and receipts 

%Inner loop
%(1) solve for value and policy functions with intial guess for pi
%(2) derive stationary distribution by iterative procedure
%(3) calculate the implied unconditional default probability from this distribution
%(4) if the percent of households choosing to default is
%sufficiently close to pi then stop; if not, increase or decrease pi 
%by half the difference between pi and the implied default rate then return
%to (1)


%%%%%%%%%%%%%%%%%%%%  Technical parameters  %%%%%%%%%%%%%%%%%%%%%%%%%
tot_outfreq   = 1;
tot_iter_tol  = 100;
tot_tol       = 1e-4;
tot_iter      = 0;
tot_dev       = 100;
tot_uptd      = 0.3;


while tot_dev > tot_tol && tot_iter < tot_iter_tol
%% Value function iteration

        iter     = 0;
        tol      = 1e-6;
        iter_tol = 1000; 
        dev      = 100;
        outfreq  = 20;


        while dev > tol && iter < iter_tol


            for ii=1:ND
               for jj=1:Ntotal  
                   for kk=1:NH
                      for ll=1:NE

                   %case (i) good standing and housed, must choose whether or not
                   %default

                   if (kk == 2 || kk==3) && ll == 1
                       %consider no default case first

                       %consumption: unhoused state
                       c0 = (1/(1+r))*D + y(jj) - D(ii);
                       %consumption: cheap housing
                       c1 = (1/(1+r))*D + y(jj) - D(ii) - R1;
                       %consumption: expensive housing
                       c2 = (1/(1+r))*D + y(jj) - D(ii) - R2;
                       %concatenate
                       cboth = [c0' c1' c2'];

                       %keep consumption positive
                       cboth(cboth<0) = 0;

                       %utility, %stays in the same  house
                       u = ((cboth.^alpha .* hval.^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                       %bellman equation 
                       bellman0 = u(:,1) + beta * oldvfunc(:,:,1,1) * Prob(jj,:)' ; %continuation value recognizes choice of homeless state
                       bellman1 = u(:,2) + beta * oldvfunc(:,:,2,1) * Prob(jj,:)' ; %continuation value recognizes choice of cheap housed state
                       bellman2 = u(:,3) + beta * oldvfunc(:,:,3,1) * Prob(jj,:)' ; %continuation value recognizes choice of expensive housed state
                       bellNoDef     = [bellman0 bellman1 bellman2];
                       %pick level of consumption, housing and debt that maximizes the
                       %bellman equation and fill into the policy functions 
                       [argpol, arg]          = max(bellNoDef(:));
                       vN(ii,jj,kk,ll)        = argpol;
                       cN(ii,jj,kk,ll)        = cboth(arg);
                       dpN(ii,jj,kk,ll)       = dval(arg);
                       hN(ii,jj,kk,ll)        = hval(arg);


                       %go on to default case

                       %consumption: housed state
                       c1 = (1/(1+r))*D + y(jj) ; %no rent paid and no assets

                       %keep consumption positive
                       c1(c1<0) = 0;

                       %utility
                       u = ((c1.^alpha .* H(kk).^(1-alpha)).^(1-sigma)-1)./(1-sigma);


                       %bellman equation
                       %tomorrow, with probability gamma you are evicted and
                       %homeless next period, with probability 1-gamma you stay in your house
                       bellDef = u' + beta * ((1-gamma) * oldvfunc(:,:,kk,2) * Prob(jj,:)' + gamma * oldvfunc(:,:,1,2) * Prob(jj,:)') ;

                       %pick level of consumption, housing and debt that maximizes the
                       %bellman equation and fill into the policy functions 
                       [argpol, arg]           = max(bellDef);
                       vD(ii,jj,kk,ll)         = argpol;
                       cD(ii,jj,kk,ll)         = c1(arg);
                       dpD(ii,jj,kk,ll)        = D(arg);
                       hD(ii,jj,kk,ll)         = H(kk);


                       %now fill in value and policy functions
                       if vN(ii,jj,kk,ll) > vD(ii,jj,kk,ll)
                           vfunc(ii,jj,kk,ll) = vN(ii,jj,kk,ll);
                           c(ii,jj,kk,ll)     = cN(ii,jj,kk,ll);
                           dp(ii,jj,kk,ll)    = dpN(ii,jj,kk,ll);
                           h(ii,jj,kk,ll)     = hN(ii,jj,kk,ll);
                           DEF(ii,jj,kk,ll)   = 0;  
                       else
                           vfunc(ii,jj,kk,ll) = vD(ii,jj,kk,ll);
                           c(ii,jj,kk,ll)     = cD(ii,jj,kk,ll);
                           dp(ii,jj,kk,ll)    = dpD(ii,jj,kk,ll);
                           h(ii,jj,kk,ll)     = hD(ii,jj,kk,ll);
                           DEF(ii,jj,kk,ll)   = 1;
                       end


                   end



                   %case (ii) good standing but unhoused, so kk = 1 and ll = 1,
                   %essentially the same as the no default case above

                   if kk == 1 && ll == 1

                       %consumption: homeless state
                       c0 = (1/(1+r))*D + y(jj) - D(ii);
                       %consumption: cheap housing state
                       c1 = (1/(1+r))*D + y(jj) - D(ii) - R1;
                       %consumption: expensive housing state
                       c2 = (1/(1+r))*D + y(jj) - D(ii) - R2;
                       %concatenate
                       cboth = [c0' c1' c2'];

                       %keep consumption positive
                       cboth(cboth<0) = 0;

                       %utility, %stays in the same  house
                       u = ((cboth.^alpha .* hval.^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                       %bellman equation 
                       bellman0 = u(:,1) + beta * oldvfunc(:,:,1,1) * Prob(jj,:)' ; %continuation value recognizes choice of homeless state
                       bellman1 = u(:,2) + beta * oldvfunc(:,:,2,1) * Prob(jj,:)' ; %continuation value recognizes choice of cheap housing state
                       bellman2 = u(:,3) + beta * oldvfunc(:,:,3,1) * Prob(jj,:)' ; %continuation value recognizes choice of expensive housing state
                       bell     = [bellman0 bellman1 bellman2];
                       %pick level of consumption, housing and debt that maximizes the
                       %bellman equation and fill into the policy functions 
                       [argpol, arg]             = max(bell(:));
                       vfunc(ii,jj,kk,ll)        = argpol;
                       c(ii,jj,kk,ll)            = cboth(arg);
                       dp(ii,jj,kk,ll)           = dval(arg);
                       h(ii,jj,kk,ll)            = hval(arg);
                   end

                   %case (iii) household defaulted last period or has been in
                   %eviction but is housed, so kk = 2 and ll = 2

                   if (kk == 2 || kk == 3) && ll == 2
                       %pays no rent but can accumulate assets

                       %consumption: housed
                       c1 = (1/(1+r))*D + y(jj) - D(ii) ; %no rent but can accumulate assets

                       %keep consumption positive
                       c1(c1<0) = 0;

                       %utility
                       u = ((c1.^alpha .* H(kk).^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                       %bellman equation 
                       bellman1 = u' + beta * ((1-gamma) * oldvfunc(:,:,kk,2) * Prob(jj,:)' + gamma * oldvfunc(:,:,1,2) * Prob(jj,:)') ;

                       %pick level of consumption, housing and debt that maximizes the
                       %bellman equation and fill into the policy functions 
                       [argpol, arg]           = max(bellman1);
                       vfunc(ii,jj,kk,ll)      = argpol;
                       c(ii,jj,kk,ll)          = c1(arg);
                       dp(ii,jj,kk,ll)         = D(arg);
                       h(ii,jj,kk,ll)          = H(kk);

                   end

                   %case (iv) household has been successfully evicted and is
                   %unhoused, waiting to be redeemed to good standing

                   if kk == 1 && ll == 2

                       %consumption: housed state
                       c1 = (1/(1+r))*D + y(jj) - D(ii) ; %no rent but can accumulate assets

                       %keep consumption positive
                       c1(c1<0) = 0;

                       %utility
                       u = ((c1.^alpha .* h0.^(1-alpha)).^(1-sigma)-1)./(1-sigma);

                       %bellman equation 
                       bellman1 = u' + beta * ((1-psi) * oldvfunc(:,:,1,2) * Prob(jj,:)' + psi * oldvfunc(:,:,1,1) * Prob(jj,:)') ;

                       %pick level of consumption, housing and debt that maximizes the
                       %bellman equation and fill into the policy functions 
                       [argpol, arg]           = max(bellman1);
                       vfunc(ii,jj,kk,ll)      = argpol;
                       c(ii,jj,kk,ll)          = c1(arg);
                       dp(ii,jj,kk,ll)         = D(arg);
                       h(ii,jj,kk,ll)          = h0;

                   end


                     end
                  end
               end
            end


            %calculate distance
            dev = max(max(max(max(abs(vfunc - oldvfunc)))));
            iter = iter + 1;    

            %simple updating
            %oldvfunc = vfunc;

            %mcqueen porteus bounds and update,
            bunder = (beta/(1 - beta)) * min(min(min(min(vfunc - oldvfunc))));
            bover  = (beta/(1 - beta)) * max(max(max(max(vfunc - oldvfunc))));
            oldvfunc = vfunc + (bunder + bover)/2;




    %         if mod(iter, outfreq) == 0
    %         fprintf('%d          %1.7f \n',iter,dev);
    %         fprintf('dp=  %1.8f  \n', min(min(min(min(dp)))))
    %         fprintf('dp=  %1.8f  \n', max(max(max(max(dp)))))
    %         fprintf('c=  %1.8f  \n', min(min(min(min(c)))))
    %         fprintf('c=  %1.8f  \n', max(max(max(max(c)))))
    %         end



        end


        fprintf('policy functions computed\n\n');
%% Stationary distribution

        iter_lambda     = 0;
        iter_tol_lambda = 1000;
        outfreq_lambda  = 50;
        dev_lambda      = 100;
        tol_lambda      = 1e-6;


        while dev_lambda > tol_lambda && iter_lambda < iter_tol_lambda

           old_lambda = lambda;

           for i=1:ND
               for j=1:Ntotal
                   for k=1:NH
                       for l=1:NE

                           value = 0;

                           for ii=1:ND
                               for jj=1:Ntotal
                                  for kk=1:NH
                                      for ll=1:NE 

                                         %case (i) if you are housed in bad
                                         %standing you cannot transition to good
                                         %standing in this model and if you are
                                         %unhoused in bad standing you must wait to
                                         %be redeemed

                                          if (k == 2 || k == 3) && l == 1

                                              if (kk == 2 || kk == 3) && ll == 1

                                                    value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i))* (h(ii,jj,kk,ll)==H(k)) * (1-DEF(ii,jj,kk));

                                              elseif kk==1 && ll==1

                                                    value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k));

                                              end
                                          end




                                          %case (ii) 

                                          if k == 1 && l == 1

                                              if kk==1 && ll==1                                        
                                                    value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k));   

                                              elseif kk==1 && ll==2
                                                  %with probability psi the agent
                                                  %transitions to good standing
                                                    value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * psi * (dp(ii,jj,kk,ll)==D(i));

                                              elseif (kk==2 || kk==3) && ll==1
                                                    value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k)); 
                                              end
                                          end

                                          %case (iii) if you are unhoused in
                                          %good standing you cannot transition to
                                          %this state, nor if you have been evicted
                                          %and are unhoused

                                          if (k == 2 || k == 3) && l == 2

                                              if (kk==2 || kk==3) && ll==2
                                                    %remain in house with
                                                    %probability 1-gamma
                                                    value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (1-gamma) * (h(ii,jj,kk,ll)==H(k)) * (dp(ii,jj,kk,ll)==D(i));

                                              elseif (kk==2 || kk==3) && ll==1
                                                    %defaulting agent moves to
                                                    %housed but bad standing with
                                                    %probability 1-gamma
                                                    value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (1-gamma) * (dp(ii,jj,kk,ll)==D(i))* (h(ii,jj,kk,ll)==H(k)) * DEF(ii,jj,kk,ll);
                                              end
                                          end



                                          %case (iv) sum only over kk=2,ll=2, kk=1, ll=2 
                                          %and kk=2,ll=1 because agent cannot go from
                                          %unhoused in good standing to unhoused in
                                          %bad standing                                  

                                          if k == 1 && l == 2

                                              if (kk==2 || kk==3) && ll==2
                                                  %agent is successfully evicted
                                                  %with probablity gamma
                                                  value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * gamma * (dp(ii,jj,kk,ll)==D(i)); 

                                              elseif kk==1 && ll==2
                                                  %agent remains unhoused in bad
                                                  %standing with probability 1-psi
                                                  value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * (1-psi) * (dp(ii,jj,kk,ll)==D(i)) * (h(ii,jj,kk,ll)==H(k));

                                              elseif (kk==2 || kk==3) && ll==1
                                                  %defaulting agent successfully
                                                  %evicted with probability gamma
                                                  value = value + old_lambda(ii,jj,kk,ll)*Prob(jj,j) * gamma * (dp(ii,jj,kk,ll)==D(i)) * DEF(ii,jj,kk,ll);                               
                                              end

                                          end



                                      end
                                  end      
                               end
                            end


            lambda(i,j,k,l) = value;


                       end
                   end
               end
           end

            %calculate distance
            dev_lambda = max(max(max(max(abs(lambda - old_lambda)))));
            iter_lambda = iter_lambda + 1; 

            if mod(iter_lambda, outfreq_lambda) == 0
            fprintf('%d          %1.7f \n',iter_lambda,dev_lambda);
            end


        end


        fprintf('\nstationary distribution computed\n\n');
%% Calculate default rates

    implied_def_prob_total = sum(sum(sum(sum(DEF.*lambda))))/(sum(sum(lambda(:,:,2,1)))+sum(sum(lambda(:,:,3,1))));

    
    omega0_1 = sum(sum(sum(lambda(:,:,2,:))));
    omega1_1 = sum(sum(DEF(:,:,2,1).*lambda(:,:,2,1)));
    omega2_1 = sum(sum(lambda(:,:,2,2)));
    
    omega0_2 = sum(sum(sum(lambda(:,:,3,:))));
    omega1_2 = sum(sum(DEF(:,:,3,1).*lambda(:,:,3,1)));
    omega2_2 = sum(sum(lambda(:,:,3,2)));
%% Update

    %calculate excess insurance payments
    xi = (tau*R1*omega0_1 + tau*R2*omega0_2)-(E*omega1_1 + R1*(omega1_1+omega2_1) + E*omega1_2 + R2*(omega1_2 + omega2_2));
    
    %calculate distance
    tot_dev = abs(xi);
    tot_iter = tot_iter + 1; 
    
    tau = tau + tot_uptd*(-1*xi);
    
    
    %equilibrium rental price
    R1 = (p1 * r)/(1-tau);
    R2 = (p2 * r)/(1-tau);
%% Summary statistics

    %average debt holding
    avedebt = sum(sum(sum(sum(dp.*lambda))));

    %percentage of time in expensive unit, paying
    time2 = sum(sum(lambda(:,:,3,1)));
    
    
    riskFreeRent1 = p1*r;
    riskFreeRent2 = p2*r;

    %percent above risk free rent
    markup = tau/(1-tau);
    
    
    if mod(tot_iter, tot_outfreq) == 0
            fprintf('%d          %1.7f \n',tot_iter,tot_dev);
            fprintf('xi  %1.8f  \n', xi)
            fprintf('tau  %1.8f  \n', tau)
            fprintf('R1  %1.8f  \n', R1)
            fprintf('R2  %1.8f  \n\n', R2)
            fprintf('Markup %1.8f \n', markup)
            fprintf('Time in 2  %1.8f  \n', time2)
            fprintf('Average debt  %1.8f  \n', avedebt)
            fprintf('Total default rate %1.8f \n\n', implied_def_prob_total)
     end
    
    
    
end
%% Save output

save('solution_withInsurance')
%% PART III: welfare analysis and plots

%housekeeping
clearvars
close all
clc



%load value function for baseline model
load solution_baseline

vB  = vfunc;
dpB = dp;
paramsB = [alpha; sigma; beta; r; p1; p2; gamma; psi; E; rho; sigma_y; h0; h1; h2];

%stationary distribution in the baseline model
lambda_B = lambda;

%load value function for the model with insurance
load solution_withInsurance

vI  = vfunc;
dpI = dp;
paramsI = [alpha; sigma; beta; r; p1; p2; gamma; psi; E; rho; sigma_y; h0; h1; h2];

%stationary distribution for the model with insurance
lambda_I = lambda;


%be sure the models are solved under the same parameterization
paramsB == paramsI



%as written, lambda is the percent increase in consumption in the baseline
%model needed to make the consumer indifferent between that economy and the
%economy with the landlord insurance scheme. Accordingly, if the insurance
%regime increases welfare, then lambda should be strictly positive


stuff = 1/((1-sigma)*(1-beta));

mu = ((vI + stuff) ./ (vB + stuff)).^(1/(alpha*(1-sigma))) - 1;


%using the baseline economy's stationary distribution: 10.58%
average_welfare_gain_B = sum(sum(sum(sum(mu.*lambda_B))));

%using the distribution under the insurance regime, gives an even better
%value but less clear interpretation: 11.06%
%average_welfare_gain_I = sum(sum(sum(sum(mu.*lambda_I))));

%now doing it the other way!!

mu2 = ((vB + stuff) ./ (vI + stuff)).^(1/(alpha*(1-sigma))) - 1;

%how much needs to be taken from household in I to make indifferent between
%living in the baseline economy, uses insurance regime stationary dist.
average_welfare_loss_I = sum(sum(sum(sum(mu2.*lambda_I))));
%% Moments of baseline model

%housekeeping
clearvars
close all
clc

%load value function for baseline model
load solution_baseline

%eviction/nonpayment rate
%data: 2.3-3.13% eviction, 6-7.5% filing
%model: 4.7%

%normalize percent nonpaying by percent of agents currently housed
eviction_rate = sum(sum(sum(sum(DEF.*lambda))))/(sum(sum(lambda(:,:,2,1)))+sum(sum(lambda(:,:,3,1))));


%percent of agents hand-to-mouth
%data (italy): 23%
%model: 69%

%find out how many agents have less than 2 weeks (1/6) of period income in
%liquid assets
htm1 = sum(sum(sum(lambda(find(D>(-1*y(1)/6)),1,:,:))));
htm2 = sum(sum(sum(lambda(find(D>(-1*y(2)/6)),2,:,:))));
htm3 = sum(sum(sum(lambda(find(D>(-1*y(3)/6)),3,:,:))));
htm4 = sum(sum(sum(lambda(find(D>(-1*y(4)/6)),4,:,:))));
htm5 = sum(sum(sum(lambda(find(D>(-1*y(5)/6)),5,:,:))));

htm_total = htm1 + htm2 + htm3 + htm4 + htm5;


%fraction of income spent on housing by those housed
%seems a bit high
%model: 55.3%


%first get marginal distribution for those who are housed
ioh1 = sum(lambda(:,:,2,1))/(sum(sum(lambda(:,:,2,1)))+sum(sum(lambda(:,:,3,1))));
ioh2 = sum(lambda(:,:,3,1))/(sum(sum(lambda(:,:,2,1)))+sum(sum(lambda(:,:,3,1))));
ioh = sum(ioh1.*(R1./y)' + ioh2.*(R2./y)');

%now checking just on the percent spent for those in the expensive units
%about 42% so still high
iohtest = sum(lambda(:,:,3,1))/sum(sum(lambda(:,:,3,1)));
iohtest = sum(iohtest.*(R2./y)');
%% Graphs

%stationary distributions for each regime: people fewer assets under this
%regime, which seems reasonable

assetsB = sum(lambda_B,[2 3 4]);
assetsI = sum(lambda_I,[2 3 4]);

figure;
hold on;
plot(D,assetsB,'ro-');
title('Stationary Distribution: $d^\prime$','interpreter','latex','FontSize',12);
xlabel('Current debt holding, $d$','FontSize',12,'interpreter','latex');
%legend({'$y$ Lowest','$y$ Mid','$y$ Highest'},'Location','SouthEast','interpreter','latex','FontSize',12);

saveas(gcf,'Figures/stationary_dist_B.png')



figure;
hold on;
plot(D,assetsI,'ro-');
title('Stationary Distribution: $d^\prime$','interpreter','latex','FontSize',12);
xlabel('Current debt holding, $d$','FontSize',12,'interpreter','latex');
%legend({'$y$ Lowest','$y$ Mid','$y$ Highest'},'Location','SouthEast','interpreter','latex','FontSize',12);

saveas(gcf,'Figures/stationary_dist_I.png')




%next period debt
figure;
hold on;
plot(D,dpB(:,1,2,1),'ro-');
plot(D,dpB(:,3,2,1),'kx-');
plot(D,dpB(:,5,2,1),'cs-');
title('Next period debt holding: $d^\prime \mid h^s=1$','interpreter','latex','FontSize',12);
xlabel('Current debt holding, $d$','FontSize',12,'interpreter','latex');
legend({'$y$ Lowest','$y$ Mid','$y$ Highest'},'Location','SouthEast','interpreter','latex','FontSize',12);

saveas(gcf,'Figures/dpB.png')

figure;
hold on;
plot(D,dpI(:,1,2,1),'ro-');
plot(D,dpI(:,3,2,1),'kx-');
plot(D,dpI(:,5,2,1),'cs-');
title('Next period debt holding: $d^\prime \mid h^s=1$','interpreter','latex','FontSize',12);
xlabel('Current debt holding, $d$','FontSize',12,'interpreter','latex');
legend({'$y$ Lowest','$y$ Mid','$y$ Highest'},'Location','SouthEast','interpreter','latex','FontSize',12);

saveas(gcf,'Figures/dpI.png')


%both regimes together
figure;
hold on;
plot(D(50:80),dpB(50:80,1,2,1),'r-');
plot(D(50:80),dpI(50:80,1,2,1),'k-');
title('Next period debt holding: $d^\prime \mid h^s=1$','interpreter','latex','FontSize',12);
xlabel('Current debt holding, $d$','FontSize',12,'interpreter','latex');
legend({'Baseline','With Insurance'},'Location','SouthEast','interpreter','latex','FontSize',12);

saveas(gcf,'Figures/dpCompare.png')



%net period debt in eviction
figure;
hold on;
plot(D,dp(:,1,2,2),'ro-');
plot(D,dp(:,3,2,2),'kx-');
plot(D,dp(:,5,2,2),'cs-');
title('Next period debt holding: $d^\prime$','interpreter','latex','FontSize',12);
xlabel('Current debt holding, $d$','FontSize',12,'interpreter','latex');
legend({'$y$ Lowest','$y$ Mid','$y$ Highest'},'Location','SouthEast','interpreter','latex','FontSize',12);

saveas(gcf,'Figures/dpB_badstanding.png')