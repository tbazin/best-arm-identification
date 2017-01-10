# Run Experiments, display results (and possibly save data) on a Bandit Problem to be specified

using HDF5

# DO YOU WANT TO SAVE RESULTS?
typeExp = "Save"
# typeExp = "NoSave"

# TYPE OF DISTRIBUTION 
@everywhere typeDistribution="Bernoulli"
@everywhere include("BAIalgos.jl")

# CHANGE NAME (save mode)
fname="results/"

# BANDIT PROBLEM
@everywhere mu=vec([0.1 0.2 0.3 0.4 0.25 0.2 0.1])
@everywhere best=find(mu.==maximum(mu))[1]
K=length(mu)

# RISK LEVEL
delta=0.1

# NUMBER OF SIMULATIONS
N=100


# OPTIMAL SOLUTION
@everywhere v,optWeights=OptimalWeights(mu)
@everywhere gammaOpt=optWeights[best]
print("mu=$(mu)\n")
print("Theoretical number of samples: $((1/v)*log(1/delta))\n")
print("Optimal weights: $(optWeights)\n\n")

# POLICIES 

@everywhere ChernoffPTSHalf(x,y,z)=ChernoffPTS(x,y,z,0.5)
@everywhere ChernoffPTSOpt(x,y,z)=ChernoffPTS(x,y,z,gammaOpt)

policies=[ATLUCB, TrackAndStop]  #, ChernoffBC2,ChernoffPTSHalf,ChernoffPTSOpt,KLLUCB,UGapE]
names=["ATLUCB", "TrackAndStop"]  #, "ChernoffBC","ChernoffPTS","ChernoffPTSOpt","KLLUCB","UGapE"]


# EXPLORATION RATES 
@everywhere explo(t,n,delta)=log((log(t)+1)/delta)

lP=length(policies)
rates=[explo for i in 1:lP]


# RUN EXPERIMENTS

function MCexp(mu,delta,N)
    for imeth=1:lP
        policy=policies[imeth]
        beta=rates[imeth]
	startTime=time()
	a = Array(RemoteRef, N)
	for j in 1:N
	    a[j] =  @spawn policy(mu,delta,beta)
	end
	res = collect([fetch(a[j]) for j in 1:N])
        proportion = zeros(N,K)
        for j in 1:N
            n=res[j][2]
            proportion[j,:]=n/sum(n)
        end
        NonTerminated = sum(collect([res[j][1]==0 for j in 1:N]))
        FracNT=(NonTerminated/N)
        FracReco=zeros(K)
        for k in 1:K
            FracReco[k]=sum([(res[j][1]==k)?1:0 for j in 1:N])/(N*(1-FracNT))
        end
        print("Results for $(policy), average on $(N) runs\n")  
	print("proportion of runs that did not terminate: $(FracNT)\n") 	 	    
	print("average number of draws: $(sum([sum(res[j][2]) for j in 1:N])/(N-NonTerminated))\n")
        print("average proportion of draws: \n $(mean(proportion,1))\n")
	print("proportion of errors: $(1-sum([res[j][1]==best for j in 1:N])/(N-NonTerminated))\n")
        print("proportion of recommendation made when termination: $(FracReco)\n")
        print("elapsed time: $(time()-startTime)\n\n")
	print("")
    end
end


function SaveData(mu,delta,N)
    K=length(mu)
    for imeth=1:lP
        Draws=zeros(N,K)
        policy=policies[imeth]
        namePol=names[imeth]
        startTime=time()
	a = Array(RemoteRef, N)
        rate=rates[imeth]
	for j in 1:N
	    a[j] =  @spawn policy(mu,delta,rate)
	end
	res = [fetch(a[j]) for j in 1:N]
        proportion=zeros(N,K)
        for k in 1:N
            r=res[k][2]
            Draws[k,:]=res[k][2]
            proportion=r/sum(r)
        end
        Reco=[res[j][1] for j in 1:N]
        Error=collect([(r==best)?0:1 for r in Reco])
        FracNT=sum([r==0 for r in Reco])/N
        FracReco=zeros(K)
        for k in 1:K
            FracReco[k]=sum([(r==k)?1:0 for r in Reco])/(N*(1-FracNT))
        end
        print("Results for $(policy), average on $(N) runs\n") 
	print("proportion of runs that did not terminate: $(FracNT)\n")  	    
	print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
        print("average proportions of draws: $(mean(proportion,1))\n")
	print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
        print("proportion of recommendation made when termination: $(FracReco)\n")
        print("elapsed time: $(time()-startTime)\n\n")
        name="$(fname)_$(namePol)_delta_$(delta)_N_$(N).h5"
        h5write(name,"mu",mu)
        h5write(name,"delta",delta)
        h5write(name,"FracNT",collect(FracNT))
        h5write(name,"FracReco",FracReco)
        h5write(name,"Draws",Draws)
        h5write(name,"Error",Error)
    end
end


function avg_diffLength!(seq1, seq2, num_sequences)
    # Average sequences of various length by extending with right-padding
    delta = length(seq2) - length(seq1)
    if delta > 0
        # Extend seq1
        append!(seq1, seq1[end]*ones(1:delta))
    elseif delta < 0
        # Extend seq2
        append!(seq2, seq2[end]*ones(1:abs(delta)))
    end

    print("Sequence 1, first elems: $(seq1[1:10])\n")
    print("Sequence 2, first elems: $(seq2[1:10])\n")
    seq1 += seq2 ./ float(num_sequences)
    return seq1
end

function multipleAVG_bestM(mu, delta, policy, namePol, N, M;
                           save=false, saveFolder="./results", overwrite=false)
    best = find(mu .== maximum(mu))[1]
    
    avg_reco_correct = Float64[0.]
    avg_min_mean = Float64[0.]
    avg_sum_means = Float64[0.]
    
    # Average over multiple independent runs
    for i in 1:N
        (_, _, reco_history, min_means_history,
         sum_means_history) = policy(mu, delta, explo, M)

        avg_reco_correct = avg_diffLength!(avg_reco_correct, reco_history, N)
        avg_min_mean = avg_diffLength!(avg_min_mean, min_means_history, N)
        avg_sum_means = avg_diffLength!(avg_sum_means, sum_means_history, N)
    end
    
    if save
        path = "$(saveFolder)/delta_$(delta)_N_$(N)_M_$(M)"
        mkpath(path)

        name = "$(path)/$(namePol).h5"
        if  overwrite && isfile(name)
            rm(name)
        end
        if !(isfile(name))
            h5write(name, "mu", mu)
            h5write(name, "delta", delta)
            h5write(name, "AverageRecommendationSuccess", avg_reco_correct)
            h5write(name, "AverageMinRecoMean", vec(avg_min_mean))
            h5write(name, "AverageSumRecoMeans", vec(avg_sum_means))
        else
            warn("Results already stored for this experiment, consider removing before updating")
        end
    end
end

function init_plotAVG()
    fig, ax_proba = subplots()
    clf()
    # plot(avg_reco_error, linewidth=1, marker="+", label=method_name)
    xlabel("Number of samples")
    ylabel(L"$P(J(t) = J^{\ast{}})$")
    title("M=$M, evolution of the success probability")
    
    fig, ax_min = subplots()
    clf()
    # plot(avg_min_mean_atlucb, linewidth=1, marker="+")
    xlabel("Number of samples")
    ylabel(L"$min_{i \in J(t)} \mu_i$")
    title("M=$M, evolution of the smallest returned mean")
    
    fig, ax_sum = subplots()
    clf()
    # plot(avg_sum_means_atlucb, linewidth=1, marker="+")
    xlabel("Number of samples")
    ylabel(L"$\Sigma_{i \in J(t)} \mu_i$")
    title("M=$M, evolution of the sum of the returned means")

    return ax_proba, ax_min, ax_sum
end

function plotAVG(ax_proba, ax_min, ax_sum, delta, N, M, namePol;
                 saveFolder="./results")
    path = "$(saveFolder)/delta_$(delta)_N_$(N)_M_$(M)"
    if isdir(path)
        fullFilePath = "$(path)/$(namePol).h5"
        if isfile("$(fullFilePath)")
            avg_reco_correct = h5read(fullFilePath, "AverageRecommendationSuccess")
            avg_min_mean = h5read(fullFilePath, "AverageMinRecoMean")
            avg_sum_means = h5read(fullFilePath, "AverageSumRecoMeans")
            
            print("Average recommendation success: $(avg_reco_correct[1:10])\n")
            
            # ax_proba[:set_xlim]([])
            figure()
            plot(avg_reco_correct[1:end], linewidth=1, marker="+",
                 label=namePol)
            figure()
            plot(avg_min_mean[1:min(1000, end)], linewidth=1, marker="+",
                 label=namePol)
            figure()
            plot(avg_sum_means[1:min(1000, end)], linewidth=1, marker="+",
                 label=namePol)
            #=
            ax_proba[:plot](avg_reco_correct, linewidth=1, marker="+",
            label=policyName)
            
            ax_min[:plot](avg_min_mean, linewidth=1, marker="+",
            label=policyName)
            
            ax_sum[:plot](avg_sum_means, linewidth=1, marker="+",
            label=policyName)
            =#
        else
            error("File $(path)/$(namePol).h5 not found")
        end
    else
        warn("No files found in directory: $(saveFolder)")
    end
end

mu = rand(10); # [0.1, 0.2, 0.5, 0.3, 0.45, 0.48]
print("Best = $(find(mu .== maximum(mu))[1])")

policies = [ATLUCB, TrackAndStop]
namePols = ["ATLUCB", "TrackAndStop"]
do_policies = [1]
N = 10
M = 2
overwrite = true
for policy in do_policies
    multipleAVG_bestM(mu, delta, policies[1], namePols[1], N, M,
                      save=true, saveFolder="./results", overwrite=overwrite)
end

ax_proba, ax_min, ax_sum = init_plotAVG()

for policy_index in do_policies
    plotAVG(ax_proba, ax_min, ax_sum, delta, N, M, namePols[policy_index],
            saveFolder="./results")
end

if false
    if (typeExp=="Save")
        SaveData(mu,delta,N)
    else
        MCexp(mu,delta,N)
    end
end
