using StochasticAD
using Statistics
using StatsBase
using Zygote
using Distributions
using DistributionsAD
using ProtoStructs
using UnPack
using ChainRulesCore

@proto struct StateSpaceModel
    T::Integer
    prior
    state_model
    obs_model
    prop_model
end

#==
note that using distributions is often slower than we would like. therefore it if typically advantageous to define a custom struct for state_model, obs_model, and prop_model, and then extend Distributions.pdf and Base.rand to each.

for example:

struct MVNR
    μ
    Σ
end

function Distributions.pdf(d::MVNR, x)
    k = length(μ)
    return 2π^(-k/2) * det(d.Σ) ^ -0.5 * exp(-0.5 * (x - d.μ)' * inv(d.Σ) * (x - d.μ))
end

function Base.rand(d::MVNR)
    k = length(μ)
    return d.μ + sqrt(d.Σ) * randn(k)
end

these functions and structs can be optimised, for example if Σ is fixed then multiple values could be precomputed in the struct
however, in the MvNormal case it is typically better to use Distributions unless Σ or μ are fixed and known
==#

@proto struct ParticleFilter
    K::Integer
    SSM::StateSpaceModel
    obs
    sample_strat
end

function generate_fake_trajectory(SSM::StateSpaceModel, θ)
    @unpack T, prior, state_model, obs_model = SSM
    x = rand(prior(θ))
    y = rand(obs_model(x,θ))
    xs = [x]
    ys = [y]
    for t in 2:T
        x = rand(state_model(x, θ))
        y = rand(obs_model(x, θ))
        push!(xs, x)
        push!(ys, y)
    end
    xs, ys
end

function strat_sample(w, K, ω = 1)
    n = length(w)
    U = rand()
    is = zeros(Int, K)
    i = 1
    cw = w[1]
    for k in 1:K
        t = ω * (k - 1 + U) / K
        while cw < t && i < n
            i += 1
            cw += w[i]
        end
        is[k] = i
    end
    return is
end

function resample(K, X, w, sample_strategy, _unw = true)
    #==
    TODO: IMPLEMENT SINKHORN ALGORITHM FOR DET RESAMPLING (should not be all too difficult)
    implement gradient stitching as well, should spped it up for freeeeeeee, but probably not needed
    ==#
    # N = size(X, 2)
    ω = sum(w)
    idx = ChainRulesCore.ignore_derivatives(() -> sample_strategy(w, K, ω)) 
    X_n = X[idx]
    if _unw
        w_c = w[idx]
        w_n = map(w -> ω .* new_weight(w ./ ω) ./ K, w_c)
    else
        w_new = fill(ω / K, K)
    end
    return X_n, w_n
end

function (F::ParticleFilter)(θ; _store = false, _unw = true, s = 1, _bpf = true)
    @unpack K, SSM, obs, sample_strat = F
    @unpack T, prior, state_model, obs_model, prop_model = SSM

    X = [rand(prior(θ)) for k in 1:K]
    w = [inv(K) for k in 1:K]
    ω = 1.

    if _store
        Xs = [X]
    end

    for (t, y) in zip(1:T, obs)
        w_o = map(x -> pdf(obs_model(x, θ), y), X)
        w = w .* w_o
        ω_o = ω
        ω = sum(w)

        if 1 < t < T && (t % s == 0) #perform resampling every s steps (placeholder)
            X, w = resample(K, X, w, sample_strat, _unw)
        end

        if t < T
            # bootstrap pf, will update to use modular proposal
            X = map(x -> rand(state_model(x, θ)), X)
            if _store
                ChainRulesCore.ignore_derivatives() do
                    push!(Xs, X)
                end
            end
        end
    end
    #if store return path, else return last state, always return weights
    return (_store ? Xs : X), w
end

function (F::ParticleFilter)(θ; _store = false, _unw = true, s = 1, _bpf = false)
    @unpack K, SSM, obs, sample_strat = F
    @unpack T, prior, state_model, obs_model, prop_model = SSM

    X = [rand(prior(θ)) for k in 1:K]
    w = [inv(K) for k in 1:K]
    ω = 1.

    ll = 0.0

    if _store
        Xs = [X]
        Ws = [w]
    end

    for (t, y) in zip(1:T, obs)

        # Normalise w_{t-1}
        w = w ./ sum(w)

        # Resample X_{t-1}
        if (t % s == 0)
            X, w = resample(K, X, w, sample_strat, _unw)
        end

        # Sample X_{t}
        X_new = map(x -> rand(prop_model(x, y, θ)), X)

        # Compute w_{t}
        t1 = map(x -> pdf(obs_model(x, θ), y), X)
        t2 = map((x, x_new) -> pdf(state_model(x, θ), x_new), X, X_new)
        t3 = map((x, x_new) -> pdf(prop_model(x, y, θ), x_new), X, X_new)
        w = t1 .* t2 ./ t3

        # Compute \hat{p}(y_{t}|y_{1:t-1})
        ω = sum(w)

        X = X_new
        if _store
            ChainRulesCore.ignore_derivatives() do
                push!(Xs, X)
                push!(Ws, w)
            end
        end

        # Accumulate \hat{l}(\theta)
        # Is technically log(sum(w_t-1 * w_t)), but w_t-1 is now uniform and normalised
        ll = ll + log(ω)
    end
    #if store return path, else return last state, always return weights
    if _store
        return Xs, Ws, ll
    else
        return X, w, ll
    end
end

function log_likelihood(F::ParticleFilter, θ, _unw = true, s = 1, _bpf = false)
    _, _, ll = F(θ, _store = false, _unw = _unw, s = s, _bpf = _bpf)
    return ll
end

function ll_grad(θ, F::ParticleFilter; s = 1, _bpf = false)
    Zygote.gradient(θ -> log_likelihood(F, θ, true, s, _bpf), θ)[1]
end
