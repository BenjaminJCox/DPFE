using StochasticAD
using Statistics
using StatsBase
using Zygote
using Distributions
using DistributionsAD
using ProtoStructs
using UnPack

@proto struct StateSpaceModel
    T::Integer
    prior
    state_model
    obs_model
end

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

function strat_sample(w, K, sw = 1)
    n = length(w)
    U = rand()
    is = zeros(Int, K)
    i = 1
    cw = w[1]
    for k in 1:K
        t = sw * (k - 1 + U) / K
        while cw < t && i < n
            i += 1
            cw += w[i]
        end
        is[k] = i
    end
    return is
end

function resample(X, w, sample_strategy, _unw = true)
    N = size(X, 2)
    ω = sum(w)
    idx = Zygote.ignore(() -> sample_strategy(w, N, ω))
    X_n = X[idx]
    if unw
        w_c = w[idx]
        w_n = map(w -> ω .* new_weight(w ./ ω) ./ N, w_c)
    else
        w_new = fill(ω / N, N)
    end
    return X_n, w_n
end

function (F::ParticleFilter)(θ; _store = false, _unw = true, s = 1)
    @unpack K, SSM, obs, sample_strat = F
    @unpack T, prior, state_model, obs_model = SSM

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
            X, w = resample(X, w, sample_strategy, _unw)
        end

        if t < T
            X = map(x -> rand(state_model(x, θ)), X)
            if _store
                Zygote.ignore(() -> push!(Xs, X))
            end
        end
    end
    #if store return path, else return last state, always return weights
    return (_store ? Xs : X), w
end

function log_likelihood(F::ParticleFilter, θ, _unw = true, s = 1)
    _, w = F(θ, _store = false, _unw = _unw, s = s)
    return log(sum(W))
end

function ll_grad(θ, F::ParticleFilter; s = 1)
    Zygote.gradient(θ -> log_likelihood(F, θ, true, s), θ)[1]
end
