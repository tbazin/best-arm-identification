# Algorithms for Best Arm Identification in Exponential Family Bandit Models in the Fixed-Confidence Setting

# The nature of the distribution should be precised by choosing a value for typeDistribution before including the current file

# All the algorithms take the following input
# mu : vector of arms means
# delta : risk level
# rate : the exploration rate (a function)

using Distributions
using PyPlot

MAX_ITERATION = 40000

include("KLfunctions.jl")

if (typeDistribution=="Bernoulli")
    d=dBernoulli
    dup=dupBernoulli
    dlow=dlowBernoulli
    function sample(mu)
        (rand()<mu)
    end
    function bdot(theta)
        exp(theta)/(1+exp(theta))
    end
    function bdotinv(mu)
        log(mu/(1-mu))
    end
elseif (typeDistribution=="Poisson")
    d=dPoisson
    dup=dupPoisson
    dlow=dlowPoisson
    function sample(mu)
        rand(Poisson(mu))       
    end
    function bdot(theta)
        exp(theta)
    end
    function bdotinv(mu)
        log(mu)
    end
elseif (typeDistribution=="Exponential")
    d=dExpo
    dup=dupExpo
    dlow=dlowExpo
    function sample(mu)
        -mu*log(rand())       
    end
    function bdot(theta)
        -log(-theta)
    end
    function bdotinv(mu)
        -exp(-mu)
    end
elseif (typeDistribution=="Gaussian")
    # sigma (std) must be defined !
    d=dGaussian
    dup=dupGaussian
    dlow=dlowGaussian
    function sample(mu)
        mu+sigma*randn()       
    end
    function bdot(theta)
        sigma^2*theta
    end
    function bdotinv(mu)
        mu/sigma^2
    end
end


# COMPUTING THE OPTIMAL WEIGHTS

function dicoSolve(f, xMin, xMax, delta=1e-11)
    # find m such that f(m)=0 using dichotomix search
    l = xMin
    u = xMax
    sgn = f(xMin)
    while u-l>delta
        m = (u+l)/2
        if f(m)*sgn>0
            l = m
        else
            u = m
        end
    end
    m = (u+l)/2
    return m
end

function I(alpha,mu1,mu2)
    if (alpha==0)|(alpha==1)
        return 0
    else
        mid=alpha*mu1 + (1-alpha)*mu2
        return alpha*d(mu1,mid)+(1-alpha)*d(mu2,mid)
    end
end

muddle(mu1, mu2, nu1, nu2) = (nu1*mu1 + nu2*mu2)/(nu1+nu2)

function cost(mu1, mu2, nu1, nu2)
    if (nu1==0)&(nu2==0)
        return 0
    else
        alpha=nu1/(nu1+nu2)
        return((nu1 + nu2)*I(alpha,mu1,mu2))
    end
end

function xkofy(y, k, mu, delta = 1e-11)
    # return x_k(y), i.e. finds x such that g_k(x)=y
    g(x)=(1+x)*cost(mu[1], mu[k], 1/(1+x), x/(1+x))-y
    xMax=1
    while g(xMax)<0
        xMax=2*xMax
    end
    return dicoSolve(x->g(x), 0, xMax, 1e-11)
end

function aux(y,mu)
    # returns F_mu(y) - 1
    K = length(mu)
    x = [xkofy(y, k, mu) for k in 2:K]
    m = [muddle(mu[1], mu[k], 1, x[k-1]) for k in 2:K]
    return (sum([d(mu[1],m[k-1])/(d(mu[k], m[k-1])) for k in 2:K])-1)
end


function oneStepOpt(mu, delta = 1e-11)
    yMax=0.5
    if d(mu[1], mu[2])==Inf
        # find yMax such that aux(yMax,mu)>0
        while aux(yMax,mu)<0
            yMax=yMax*2
        end
    else
        yMax=d(mu[1],mu[2])
    end
    y = dicoSolve(y->aux(y, mu), 0, yMax, delta)
    x =[xkofy(y, k, mu, delta) for k in 2:length(mu)]
    unshift!(x, 1)
    nuOpt = x/sum(x)
    return nuOpt[1]*y, nuOpt
end


function OptimalWeights(mu, delta=1e-11)
    # returns T*(mu) and w*(mu)
    K=length(mu)
    IndMax=find(mu.==maximum(mu))
    L=length(IndMax)
    if (L>1)
        # multiple optimal arms
        vOpt=zeros(1,K)
        vOpt[IndMax]=1/L
        return 0,vOpt
    else
        mu=vec(mu)
        index=sortperm(mu,rev=true)
        mu=mu[index] 
        unsorted=vec(collect(1:K))
        invindex=zeros(Int,K)
        invindex[index]=unsorted 
        # one-step optim
        vOpt,NuOpt=oneStepOpt(mu,delta)
        # back to good ordering
        nuOpt=NuOpt[invindex]
        NuOpt=zeros(1,K)
        NuOpt[1,:]=nuOpt
        return vOpt,NuOpt
    end
end



# OPTIMAL ALGORITHMS

function checkRecommendation(true_means, empirical_means, M)
    true_best_m = sortperm(vec(true_means), rev=true)[1:M]
    empirical_best_m = sortperm(vec(empirical_means), rev=true)[1:M]
    return sort(true_best_m) == sort(empirical_best_m)
end


function TrackAndStop(mu, delta, rate; M=1, useStopppingStatistic=true)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1

    recommendation_history = Int[]
    min_means_history = Float64[]
    sum_means_history = Float64[]
    
    while (condition)
        Mu=S./N
        # Empirical best arm
        IndMax=find(Mu.==maximum(Mu))
        Best=IndMax[floor(Int,length(IndMax)*rand())+1]

        correct_reco = checkRecommendation(mu, Mu, M)
        push!(recommendation_history, correct_reco)

        m_best_empirical_means_ind = sortperm(vec(Mu), rev=true)[1:M]
        m_best_means = mu[m_best_empirical_means_ind]
        push!(min_means_history, minimum(m_best_means)) 
        push!(sum_means_history, sum(m_best_means)) 
        
        I=1
        if (length(IndMax)>1)
            # if multiple maxima, draw one them at random 
            I = Best
        else 
       	    # compute the stopping statistic
       	    NB=N[Best]
            SB=S[Best]
            muB=SB/NB
            MuMid=(SB+S)./(NB+N)
            Index=collect(1:K)
            splice!(Index,Best)
            Score=minimum([NB*d(muB,MuMid[i])+N[i]*d(Mu[i],MuMid[i]) for i in Index])
            if (Score > rate(t,0,delta)) && useStopppingStatistic
                # stop 
                condition=false
            elseif (t > MAX_ITERATION || t > 10000000)
                # stop and outputs (0,0) 
                condition=false
                Best=0
                print(N)
                print(S)
                N=zeros(1,K) 
            else 
                if (minimum(N) <= max(sqrt(t) - K/2,0))
                    # forced exploration
                    I=indmin(N)
                else
                    # continue and sample an arm
	            val,Dist=OptimalWeights(Mu,1e-11)
                    # choice of the arm
                    I=indmax(Dist-N/t)
                end 
	    end
        end
        # draw the arm 
        t+=1
        S[I]+=sample(mu[I])
        N[I]+=1
    end
    recommendation=Best
    return (recommendation, N, recommendation_history,
            min_means_history, sum_means_history)
end


function TrackAndStop2(mu,delta,rate)
    # Uses a Tracking of the cummulated sum
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    SumWeights=ones(1,K)/K
    while (condition)
        Mu=S./N
        # Empirical best arm
        IndMax=find(Mu.==maximum(Mu))
        Best=IndMax[floor(Int,length(IndMax)*rand())+1]
        I=1
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        muB=SB/NB
        MuMid=(SB+S)./(NB+N)
        Index=collect(1:K)
        splice!(Index,Best)
        Score=minimum([NB*d(muB,MuMid[i])+N[i]*d(Mu[i],MuMid[i]) for i in Index])
        if (Score > rate(t,0,delta))
            # stop 
            condition=false
        elseif (t >1000000)
            # stop and output (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K) 
        else 
            # continue and sample an arm
	    val,Dist=OptimalWeights(Mu,1e-11)
            SumWeights=SumWeights+Dist 
	    # choice of the arm
            if (minimum(N) <= max(sqrt(t) - K/2,0))
                # forced exploration
                I=indmin(N)
            else 
                I=indmax(SumWeights-N)
            end
        end
        # draw the arm 
        t+=1
        S[I]+=sample(mu[I])
        N[I]+=1
    end
    recommendation=Best
    return (recommendation,N)
end


# Chernoff stopping rule coupled with different sampling rules 


function ChernoffTarget(mu,delta,rate,Target=ones(1,length(mu))/length(mu))
    # sampling rule : choose arm maximizing (Target - empirical proportion)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    while (condition)
        Mu=S./N
        Ind=find(Mu.==maximum(Mu))
        # Empirical best arm
        Best=Ind[floor(Int,length(Ind)*rand())+1]
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        muB=SB/NB
        MuMid=(SB+S)./(NB+N)
        Index=collect(1:K)
        splice!(Index,Best)
        Score=minimum([NB*d(muB,MuMid[i])+N[i]*d(Mu[i],MuMid[i]) for i in Index])
        if (Score > rate(t,0,delta))
            # stop 
            condition=false
        elseif (t >1000000)
            # stop and return (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K) 
        else 
	    I=indmax(Target-N/t) 
	    t+=1
	    S[I]+=sample(mu[I])
	    N[I]+=1
	end
    end
    recommendation=Best
    return (recommendation,N)
end



function ChernoffBC(mu,delta,rate)
    # Chernoff stopping rule,  sampling based on the "best challenger"
    # described in experimental section of [Garivier and Kaufmann 2016]
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    while (condition)
        Mu=S./N
        # Empirical best arm
        IndMax=find(Mu.==maximum(Mu))
        Best=IndMax[floor(Int,length(IndMax)*rand())+1]
        I=1
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        MuB=SB/NB
        MuMid=(SB+S)./(NB+N)
        Challenger=1
        Score=Inf
        for i=1:K
	    if i!=Best
                score=NB*d(MuB,MuMid[i])+N[i]*d(Mu[i],MuMid[i])
	        if (score<Score)
		    Challenger=i
		    Score=score
	        end
	    end
        end
        if (Score > rate(t,0,delta))
            # stop 
            condition=false
        elseif (t >1000000)
            # stop and return (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K) 
        else 
            # continue and sample an arm
	    val,Dist=OptimalWeights(Mu,1e-11)
            if (minimum(N) <= max(sqrt(t) - K/2,0))
                # forced exploration
                I=indmin(N)
            else 
                # choose between the arm and its Challenger 
                I=(NB/(NB+N[Challenger]) < Dist[Best]/(Dist[Best]+Dist[Challenger]))?Best:Challenger
                #I=(d(MuB,MuMid[Challenger])>d(Mu[Challenger],MuMid[Challenger]))?Best:Challenger
                #I=(N[Best]<N[Challenger])?Best:Challenger
            end
        end
        # draw the arm 
        t+=1
        S[I]+=sample(mu[I])
        N[I]+=1
    end
    recommendation=Best
    return (recommendation,N)
end



function ChernoffBC2(mu,delta,rate)
    # Chernoff stopping rule + alternative choice between the empirical best and its "challenger"
    # Faster, requires no computation of Optimal Weights
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    while (condition)
        Mu=S./N
        # Empirical best arm
        IndMax=find(Mu.==maximum(Mu))
        Best=IndMax[floor(Int,length(IndMax)*rand())+1]
        I=1
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        MuB=SB/NB
        MuMid=(SB+S)./(NB+N)
        Challenger=1
        Score=Inf
        for i=1:K
	    if i!=Best
                score=NB*d(MuB,MuMid[i])+N[i]*d(Mu[i],MuMid[i])
	        if (score<Score)
		    Challenger=i
		    Score=score
	        end
	    end
        end
        if (Score > rate(t,0,delta))
            # stop 
            condition=false
        elseif (t >1000000)
            # stop and return (0,0)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K) 
        else 
            # continue and sample an arm
	    if (minimum(N) <= max(sqrt(t) - K/2,0))
                # forced exploration
                I=indmin(N)
            else 
                # choose between the arm and its Challenger
                I=(N[Best]<N[Challenger])?Best:Challenger
                #I=(d(MuB,MuMid[Challenger])>d(Mu[Challenger],MuMid[Challenger]))?Best:Challenger
            end
        end
        # draw the arm 
        t+=1
        S[I]+=sample(mu[I])
        N[I]+=1
    end
    recommendation=Best
    return (recommendation,N)
end


function ChernoffKLLUCB(mu,delta,rate)
    # Chernoff stopping rule, KL-LUCB sampling rule 
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    while (condition)
        Mu=S./N
        Ind=find(Mu.==maximum(Mu))
        # Empirical best arm
        Best=Ind[round(Int,floor(length(Ind)*rand())+1)]	
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        muB=SB/NB
        MuMid=(SB+S)./(NB+N)
        Index=collect(1:K)
        splice!(Index,Best)
        Score=minimum([NB*d(muB,MuMid[i])+N[i]*d(Mu[i],MuMid[i]) for i in Index])	
        # Find the challenger
        UCB=zeros(1,K)
        LCB=dlow(Mu[Best],rate(t,0,delta)/N[Best])
        for a in 1:K
	    if a!=Best
	        UCB[a]=dup(Mu[a],rate(t,0,delta)/N[a])
            end
        end
        Ind=find(UCB.==maximum(UCB))
        Challenger=Ind[round(Int,floor(length(Ind)*rand())+1)]
        # draw both arms  
        t=t+2
        S[Best]+=sample(mu[Best])
        N[Best]+=1
        S[Challenger]+=sample(mu[Challenger])
        N[Challenger]+=1
        # check stopping condition
        condition=(Score <= rate(t,0,delta))
        if (t>1000000)
	    condition=false
            Best=0
            N=zeros(1,K)
        end
    end
    recommendation=Best
    return (recommendation,N)
end





# KL-LUCB [Kaufmann and Kalyanakrishnan 2013]

function KLLUCB(mu,delta,rate)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    while (condition)
        Mu=S./N
        Ind=find(Mu.==maximum(Mu))
        # Empirical best arm
        Best=Ind[round(Int,floor(length(Ind)*rand())+1)]		
        # Find the challenger
        UCB=zeros(1,K)
        LCB=dlow(Mu[Best],rate(t,N[Best],delta)/N[Best])
        for a in 1:K
	    if a!=Best
	        UCB[a]=dup(Mu[a],rate(t,N[a],delta)/N[a])
            end
        end
        Ind=find(UCB.==maximum(UCB))
        Challenger=Ind[round(Int,floor(length(Ind)*rand())+1)]
        # draw both arms  
        t=t+2
        S[Best]+=sample(mu[Best])
        N[Best]+=1
        S[Challenger]+=sample(mu[Challenger])
        N[Challenger]+=1
        # check stopping condition
        condition=(LCB < UCB[Challenger])
        if (t>1000000)
	    condition=false
            Best=0
            N=zeros(1,K)
        end 
    end
    recommendation=Best
    return (recommendation,N)
end





# Racing algorithms

function ChernoffRacing(mu,delta,rate)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    round=1
    t=K
    Best=1
    Remaining=collect(1:K)
    while (length(Remaining)>1)
        # Drawn all remaining arms 
        for a in Remaining 
	    S[a]+=sample(mu[a])
	    N[a]+=1 	
        end
        round+=1
        t+=length(Remaining)
        # Check whether the worst should be removed    
        Mu=S./N
        MuR=Mu[Remaining]
        MuBest=maximum(MuR)
        IndBest=find(MuR.==MuBest)[1]
        IndBest=IndBest[floor(Int,rand()*length(IndBest))+1]
        Best=Remaining[IndBest]
        MuWorst=minimum(MuR)
        IndWorst=find(MuR.==MuWorst)[1]
        IndWorst=IndWorst[floor(Int,rand()*length(IndWorst))+1]
        if (round*(d(MuBest, (MuBest+MuWorst)/2)+d(MuWorst,(MuBest+MuWorst)/2)) > rate(t,0,delta))
            # remove Worst arm
            splice!(Remaining,IndWorst)
        end
        if (t>1000000)
	    Remaining=[]
            Best=0
            N=zeros(1,K)
        end
    end
    recommendation=Best
    return (recommendation,N)
end



function KLRacing(mu,delta,rate)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    round=1
    t=K
    Best=1
    Remaining=collect(1:K)
    while (length(Remaining)>1)
        # Drawn all remaining arms 
        for a in Remaining 
	    S[a]+=sample(mu[a])
	    N[a]+=1 	
        end
        round+=1
        t+=length(Remaining)
        # Check whether the worst should be removed    
        Mu=S./N
        MuR=Mu[Remaining]
        MuBest=maximum(MuR)
        IndBest=find(MuR.==MuBest)[1]
        Best=IndBest[floor(Int,rand()*length(IndBest))+1]
        Best=Remaining[Best]
        MuWorst=minimum(MuR)
        IndWorst=find(MuR.==MuWorst)[1]
        IndWorst=IndWorst[floor(Int,rand()*length(IndWorst))+1]
        if (dlow(MuBest,rate(t,round,delta)/round) > dup(MuWorst,rate(t,round,delta)/round))
            # remove Worst arm
            splice!(Remaining,IndWorst)
        end
        if (t>1000000)
	    Remaining=[]
            Best=0
            N=zeros(1,K)
        end
    end
    recommendation=Best
    return (recommendation,N)
end


# UGapE [Gabillon et al., 2012]

function UGapE(mu,delta,rate)
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    while (condition)
        Mu=S./N
        Ind=find(Mu.==maximum(Mu))
        # Empirical best arm
        Best=Ind[round(Int,floor(length(Ind)*rand())+1)]		
        # Find the challenger
        UCB=zeros(1,K)
        LCB=zeros(1,K)
        for a in 1:K
            UCB[a]=dup(Mu[a],rate(t,N[a],delta)/N[a])
            LCB[a]=dlow(Mu[a],rate(t,N[a],delta)/N[a])
        end
        B=zeros(1,K)
        for a in 1:K
            Index=collect(1:K)
            splice!(Index,a)
            B[a] = maximum(UCB[Index])-LCB[a]
        end 
        Value=minimum(B)
        Best=indmin(B)
        UCB[Best]=0
        Challenger=indmax(UCB)
        # choose which arm to draw   
        t=t+1
        I=(N[Best]<N[Challenger])?Best:Challenger        
        S[I]+=sample(mu[I])
        N[I]+=1
        # check stopping condition
        condition=(Value > 0)
        if (t>1000000)
	    condition=false
            Best=0
            N=zeros(1,K)
        end 
    end
    recommendation=Best
    return (recommendation,N)
end


# Pure-Exploration Thompson Sampling [Russo, 2016] + Chernoff Stopping rule 

function ChernoffPTS(mu,delta,rate,frac,alpha=1,beta=1)
    # Chernoff stopping rule combined with the PTS sampling rule
    condition = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a]=1
        S[a]=sample(mu[a]) 
    end
    t=K
    Best=1
    while (condition)
        Mu=S./N
        Ind=find(Mu.==maximum(Mu))
        # Empirical best arm
        Best=Ind[floor(Int,length(Ind)*rand())+1]
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        muB=SB/NB
        MuMid=(SB+S)./(NB+N)
        Index=collect(1:K)
        splice!(Index,Best)
        Score=minimum([NB*d(muB,MuMid[i])+N[i]*d(Mu[i],MuMid[i]) for i in Index])
        if (Score > rate(t,0,delta))
            # stop 
            condition=false
        elseif (t >1000000)
            condition=false
            Best=0
            print(N)
            print(S)
            N=zeros(1,K) 
        else 
            TS=zeros(K)
            for a=1:K
	        TS[a]=rand(Beta(alpha+S[a], beta+N[a]-S[a]), 1)[1]
            end
            I = indmax(TS)
            if (rand()>frac)
                J=I
                condition=true
                while (I==J)
                    TS=zeros(K)
                    for a=1:K
	                TS[a]=rand(Beta(alpha+S[a], beta+N[a]-S[a]), 1)[1]
                    end
                    J= indmax(TS)
                end
                I=J
            end
            # draw arm I
	    t+=1
	    S[I]+=sample(mu[I])
	    N[I]+=1
	end
    end
    recommendation=Best
    return (recommendation,N)
end



# AT-LUCB

function bestarm(means)
    IndMax=find(means.==maximum(means))
    # Shuffle to avoid focusing on a single arm
    return shuffle(IndMax)
end

function mbestarms(means, M)
    m_best = sortperm(means, rev=true)[1:M]
    return m_best
end

function checkListEquality(true_best, recommendation)
    return sort(true_best) == sort(recommendation)
end

function ATLUCB(means, deltaStop, rate, M=1, delta1=1/2, alpha=0.99, eps=0,
                useStopppingStatistic=false)
    #=Anytime-LUCB algorithm
    
    Return a Generator object allowing to control the number of iteration
    i.e. the degree of precision of the algorithm.

    Perform successive "stages" of LUCB with decreasing failure-rate
    delta (where the failure rate is the probability of not correctly
    predicting the actual M-best arms).
    The algorithm starts at stage 1 with failure-rate delta1, and
    runs LUCB with terminating criterion Term(delta1, eps).
    This terminating criterion in LUCB ensures that the M output arms
    all have means higher than (mu_star - eps) with probability at least
    (1 - delta1).

    The algorithm is then re-run with a tighter failure-rate.

    Input arguments:
      * means, an array of floats: the Multi-Arm Bandit to estimate,
      * M, an integer: the number of best arms to track,
      * delta1, a float (in [1/200, mab.K]): the initial failure rate for
       the first stage,
      * alpha, a float (in [1/50, 1)): the discount factor, defines the
       rate at which the failure-rate decreases over stages,
      * eps, a float (with eps > 0): the terminating threshold for each stage.
    =#
    K = length(means)
    sums = zeros(K)
    drawcounts = zeros(K)

    true_best_m = mbestarms(means, M)

    # sample once from all of the arms
    for arm in 1:K
        sample_value = sample(means[arm])
        drawcounts[arm] += 1
        sums[arm] += sample_value
    end
    empirical_means = sums ./ drawcounts
    current_best_arms = mbestarms(means, M)
    
    t = 1
    stage = 1

    J = []
    push!(J, current_best_arms)

    recommendation_history = Int[]
    correct_reco = checkListEquality(true_best_m, J[t])
    push!(recommendation_history, correct_reco)
    
    min_means_history = Float64[]
    sum_means_history = Float64[]

    means_history = Float64[]
    m_best_empirical_means_ind = sortperm(empirical_means, rev=true)[1:M]
    m_best_means = means[m_best_empirical_means_ind]
    push!(min_means_history, minimum(m_best_means)) 
    push!(sum_means_history, sum(m_best_means))

    function delta_s(s)
        return delta1 * alpha^(s-1)
    end
    
    while t < MAX_ITERATION
        delta = delta_s(stage)
        
        empirical_means = sums ./ drawcounts
        current_best_arm = bestarm(empirical_means)[1]
        current_best_arms = mbestarms(empirical_means, M)
        
        if useStopppingStatistic
            # compute the stopping statistic from TrackAndStop
            NB = drawcounts[current_best_arm]
            SB = sums[current_best_arm]
            muB = SB/NB
            MuMid = (SB+sums)./(NB+drawcounts)
            Index = collect(1:K)
            splice!(Index,current_best_arm)
            Score = minimum([(NB*d(muB,MuMid[i])+
                              drawcounts[i]*d(empirical_means[i],MuMid[i]))
                             for i in Index])
            if (Score > rate(t,0,deltaStop))
                # stop
                break
            end
        end
        
        if term(empirical_means, drawcounts, M, t, delta, eps)
            min_stage = delta + 1
            new_stage = maxStageIndex(empirical_means, drawcounts,
                                      M, min_stage,
                                      delta_s, t, eps)
            if new_stage >= 100
                break
            end
            
            stage = new_stage
            push!(J, current_best_arms)
        else
            if stage == 1
                push!(J, current_best_arms)
            else
                push!(J, J[t])  # keep last estimate
            end
        end
        
        lowest_best_arm = h(empirical_means, drawcounts, M,
                            t, delta_s(stage))
        highest_bad_arm = l(empirical_means, drawcounts, M,
                            t, delta_s(stage))
        
        # Draw lowest best arm
        sample_value = sample(means[lowest_best_arm])
        drawcounts[lowest_best_arm] += 1
        sums[lowest_best_arm] += sample_value

        # store current estimates
        empirical_means = sums ./ drawcounts

        correct_reco = checkListEquality(true_best_m, J[t])
        push!(recommendation_history, correct_reco)
        
        m_best_empirical_means_ind = sortperm(empirical_means, rev=true)[1:M]
        m_best_means = means[m_best_empirical_means_ind]
        push!(min_means_history, minimum(m_best_means)) 
        push!(sum_means_history, sum(m_best_means))

        # Draw highest bad arm
        sample_value = sample(means[highest_bad_arm])
        drawcounts[highest_bad_arm] += 1
        sums[highest_bad_arm] += sample_value
        
        # store current estimates
        empirical_means = sums ./ drawcounts

        correct_reco = checkListEquality(true_best_m, J[t])
        push!(recommendation_history, correct_reco)
        
        m_best_empirical_means_ind = sortperm(empirical_means, rev=true)[1:M]
        m_best_means = means[m_best_empirical_means_ind]
        push!(min_means_history, minimum(m_best_means)) 
        push!(sum_means_history, sum(m_best_means))

        t += 1
    end
print("\nMaximum iteration reached for AT-LUCB, J[t] = $(J[t])\n")
return (J[end], drawcounts, recommendation_history,
        min_means_history, sum_means_history)
end


function maxStageIndex(empirical_means, drawcounts, M,
                       min_stage_index, delta_s_func, t, eps)
    # Get greatest stage index that is not terminating
    # NOTE: Hack here to ensure termination
    stage_index = min_stage_index
    while (!term(empirical_means, drawcounts, M,
                 t, delta_s_func(stage_index+1), eps) &&
           stage_index < 50)
        stage_index += 1
    end
    
    return stage_index
end


function checkparameters(K, delta, alpha, eps)
    @assert delta >= 1/200; @assert delta <= K
    @assert alpha >= 1/50; @assert alpha < 1
    @assert eps >= 0
end


function term(empirical_means, drawcounts, M, t, delta, eps)
    # Check if the current confidence gap is below the threshold eps
    lowest_best_arm = h(empirical_means, drawcounts, M, t, delta)
    highest_worst_arm = l(empirical_means, drawcounts, M, t, delta)
    lower_bounds = L(empirical_means, drawcounts, t, delta)
    upper_bounds = U(empirical_means, drawcounts, t, delta)

    if ((lowest_best_arm >= 0) & (highest_worst_arm  >= 0))
        confidence_gap = (upper_bounds[highest_worst_arm] -
                          lower_bounds[lowest_best_arm])
    else
        confidence_gap = Inf
    end
    
    return confidence_gap < eps
end


function beta(u, K, t, delta)
    #= Deviation function
    
    Input arguments:
    * u, a float: the current number of draws for the considered
    arm,
    * K, an int: the number of arms of the considered MAB,
    * t, an int: the current step in the AT-LUCB algorithm,
    * delta, a float: the failure-rate.
    =#
    k1 = 5/4  # constant defined in the original paper
    if u == 0
        return Inf
    else
        return sqrt(1/(2*u) * log(k1 * K * (t^4)/delta))
    end
end
    
function L(empirical_means, drawcounts, t, delta)
    # Lower confidence bound
    K = length(empirical_means)
    
    return empirical_means .- map(u -> beta(u, K, t, delta),
                                  drawcounts)
end

function U(empirical_means, drawcounts, t, delta)
    # Upper confidence bound
    K = length(empirical_means)
    
    return empirical_means .+ map(u -> beta(u, K, t, delta),
                                  drawcounts)
end


function h(empirical_means, drawcounts, M, t, delta)
    #= Smallest of the M best arms under LCB

    Arms with equal means are shuffled so as to avoid focusing on
    a single value.
    =#
    bestarms = mbestarms(empirical_means, M)

    L_best = L(empirical_means, drawcounts, t, delta)[bestarms]

    if !(isempty(L_best))
        min_best = find(L_best.==minimum(L_best))[1]
        return bestarms[min_best]
    else
        return rand(1:K)
    end
end


function l(empirical_means, drawcounts, M, t, delta)
    #= Largest of the non-top M arms under UCB

    Arms with equal means are shuffled so as to avoid focusing on
    a single value.
    =#
    K = length(empirical_means)
    bestarms = mbestarms(empirical_means, M)

    # Get the arms not in the current top M estimate
    is_not_in_bestarms = x -> !(in(x, bestarms))
    non_bestarms = filter(is_not_in_bestarms, 1:K)

    U_non_best = U(empirical_means, drawcounts, t, delta)[non_bestarms]
    
    if !(isempty(U_non_best))
        max_non_best = find(U_non_best .== maximum(U_non_best))[1]
        return non_bestarms[max_non_best]
    else
        return rand(1:K)
    end
end
