using StochasticAD
using Statistics
using StatsBase
using Zygote
using ForwardDiff
using FiniteDiff
# using Enzyme
using Distributions
using DistributionsAD
using ProtoStructs
using UnPack
using ChainRulesCore

struct StateSpaceModel
    T::Integer
    prior::Function
    state_model::Function
    obs_model::Function
    prop_model::Function
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

struct ParticleFilter
    K::Integer
    SSM::StateSpaceModel
    obs::Vector
    sample_strat::Function
end

function generate_fake_trajectory(SSM::StateSpaceModel, θ)
    @unpack T, prior, state_model, obs_model = SSM
    x = rand(prior(θ))
    y = rand(obs_model(x,θ))
    xs = []
    ys = []
    for t in 1:T
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
        # new_weight is the important function here, allows us to propagate gradients through resampling
        w_n = map(w -> ω .* new_weight(w ./ ω) ./ K, w_c)
    else
        w_n = fill(ω / K, K)
    end
    return X_n, w_n
end

function soft_resample(K, X, w, sample_strategy; α = 0.5)
    ω = sum(w)
    q_k = α .* w .+ (1-α) .* 1/K
    idx = ChainRulesCore.ignore_derivatives(() -> sample_strategy(q_k, K, ω))
    X_n = X[idx]
    w_n = w[idx] ./ q_k[idx]
    w_n = ω .* w_n ./ sum(w_n)
    return X_n, w_n
end

# function (F::ParticleFilter)(θ; _store = false, _unw = true, s = 1, _bpf = true)
#     @unpack K, SSM, obs, sample_strat = F
#     @unpack T, prior, state_model, obs_model, prop_model = SSM
#
#     X = [rand(prior(θ)) for k in 1:K]
#     w = [inv(K) for k in 1:K]
#     ω = 1.
#     ll = 0.0
#
#     if _store
#         Xs = [X]
#         Ws = [w]
#     end
#
#     for (t, y) in zip(1:T, obs)
#
#         X = map(x -> rand(state_model(x, θ)), X)
#         if _store
#             ChainRulesCore.ignore_derivatives() do
#                 push!(Xs, X)
#             end
#         end
#
#         w = map(x -> pdf(obs_model(x, θ), y), X)
#         ω = sum(w)
#         w = w ./ ω
#
#         ChainRulesCore.ignore_derivatives() do
#             @info "PRE Resampling itr $t"
#             display(summarystats(w))
#         end
#
#         if t % s == 0 #perform resampling every s steps (placeholder)
#             X, w = resample(K, X, w, sample_strat, _unw)
#         end
#         ChainRulesCore.ignore_derivatives() do
#             @info "POST Resampling itr $t"
#             display(summarystats(w))
#         end
#
#         ll = ll + log(ω)
#     end
#     #if store return path, else return last state, always return weights
#     if _store
#         @info "DING"
#         return Xs, Ws, ll
#     else
#         return X, w, ll
#     end
# end

function (F::ParticleFilter)(θ; _store = false, _unw = true)
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

        # See SARKAA pg 195-196
        # Sample X_{t}
        X_new = map(x -> rand(prop_model(x, y, θ)), X)

        # Compute w_{t}
        t1 = map(x -> pdf(obs_model(x, θ), y), X_new)
        t2 = map((x, x_new) -> pdf(state_model(x, θ), x_new), X, X_new)
        t3 = map((x, x_new) -> pdf(prop_model(x, y, θ), x_new), X, X_new)
        w = t1 .* t2 ./ t3 .* w

        # Compute \hat{p}(y_{t}|y_{1:t-1})
        ω = sum(w)

        # Accumulate \hat{l}(\theta)
        # Is technically log(sum(w_t-1 * w_t)), but w_t-1 is now uniform and normalised
        # see SARKAA PG 196: p(y_{1:T}|θ) ≈ ∏\hat{p}(y_{t}|y_{1:t-1}, θ)
        # where \hat{p}(y_{k}|y_{1:k-1}, θ) = ∑\bar{w}_{t-1}w_t
        # if just calculate sum of final weights we do NOT get an estimate of likelihood  p(y_{1:T}|θ) as desired
        ll = ll + log(ω)

        # As we no longer need the old path we now update and push to storage
        X = X_new
        if _store
            ChainRulesCore.ignore_derivatives() do
                push!(Xs, X)
                push!(Ws, w)
            end
        end

        # normalise weights for resampling
        w = w ./ ω
        # Resample X_{t}
        # this sets w to 1/K
        # X, w = resample(K, X, w, sample_strat, _unw)
        X, w = soft_resample(K, X, w, sample_strat, α = 0.5)

    end
    #if store return path, else return last state, always return weights
    if _store
        return Xs, Ws, ll
    else
        return X, w, ll
    end
end

function log_likelihood(F::ParticleFilter, θ, _unw = true)
    _, _, ll = F(θ, _store = false, _unw = _unw)
    return ll
end

function energy(F::ParticleFilter, θ, pθ::Distribution, _unw = true)
    _, _, ll = F(θ, _store = false, _unw = _unw)
    return logpdf(pθ, θ) .+ ll
end

#==
Gradient helpers
AS THE FUNCTION IS STOCHASTIC THESE GRADIENTS ARE DRAWS FROM THE DISTRIBUTION OF THE GRADIENT AT θ
ONE MUST BE CAREFUL USING THEM AS NaNs ARE TO BE EXPECTED IF θ IS SENSITIVE:
E.G IF θ>1 YIELDS UNSTABLE SYSTEM, THEN NEED TO TAKE SMALLER STEPS FOR θ CLOSE TO 1 AS LIKELIHOOD ESTIMATES MAY(WILL) EXPLODE
REMEMBER IT IS POSSIBLE AND ALLOWABLE TO AVERAGE THESE GRADIENTS OVER MULTIPLE EVALUATIONS OF THE SYSTEM
if system is derived from differential equation consider performing a stability analysis beforehand
==#

# log likelihood
function ll_grad_zyg(θ, F::ParticleFilter)
    Zygote.gradient(θ -> log_likelihood(F, θ, true), θ)[1]
end

# function ll_grad_enz(θ, F::ParticleFilter)
#     _dθ = zero(θ)
#     Enzyme.autodiff(Enzyme.Reverse, θ -> log_likelihood(F, θ, true), Active, Duplicated(θ, _dθ))
#     return _dθ
# end
# errors with abstract type integer does not have a definite size, no idea how to fix

function ll_grad_fwd(θ, F::ParticleFilter)
    ForwardDiff.gradient(θ -> log_likelihood(F, θ, true), θ)
end

function ll_hess_fwd(θ, F::ParticleFilter)
    ForwardDiff.hessian(θ -> log_likelihood(F, θ, true), θ)
end

# energy (sometimes better)
function en_grad_zyg(θ, pθ, F::ParticleFilter)
    Zygote.gradient(θ -> energy(F, θ, pθ, true), θ)[1]
end

function en_grad_fwd(θ, pθ, F::ParticleFilter)
    ForwardDiff.gradient(θ -> energy(F, θ, pθ, true), θ)
end

function en_hess_fwd(θ, pθ, F::ParticleFilter)
    ForwardDiff.hessian(θ -> energy(F, θ, pθ, true), θ)
end
