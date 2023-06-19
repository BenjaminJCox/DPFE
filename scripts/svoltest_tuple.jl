using DrWatson
using Random
using LinearAlgebra
using CairoMakie

include("../src/dpf_stochad.jl")

#==

==#

Random.seed!(12)

prior(θ) = Normal(0., sqrt(θ[3]^2 / (1-θ[1]^2)))

f_dyn(x, θ) = θ[1] .* x

g_obs(x, θ) = θ[2] .* exp.(x ./ 2)

state_model(x, θ) = Normal(f_dyn(x, θ)[1], θ[3])
obs_model(x, θ) = Normal(0.0, θ[2] .* exp.(x[1]/2))

Zygote.gradient(θ -> mean(rand(state_model([0.5], θ))), [0.91, 0.5, 1.0])

prop_model(x, y, θ) = Normal(f_dyn(x, θ)[1], 2.) # not quite bootstrap pf

T = 500

θ = (0.91, 0.5, 1.0)
#[α, β, σ], parameters and parameterisation from https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
# note that the parametrisation given in the tutorial is N(μ, σ^2), whereas the parametrisation here is N(μ, σ)
# this is important for the observation model

test_SSM = StateSpaceModel(T, prior, state_model, obs_model, prop_model)

test_xs, test_ys = generate_fake_trajectory(test_SSM, θ)

p_x1s = getindex.(test_xs, 1)

pfig = Figure()
pax = Axis(pfig[1, 1])

scatter!(pax, collect(1:T), p_x1s)

pfig

K = 100


test_pf = ParticleFilter(K, test_SSM, test_ys, strat_sample)

Random.seed!(12)
tpfo = test_pf(θ, _store = true)

Random.seed!(1)
_tgzg = ll_grad_zyg(θ, test_pf)

msx = [0. for i in 1:T]

for i in 1:K
    local xs = [tpfo[1][t][i] for t in 1:T]
    msx .+= xs
end

msx ./= K


# _vov = tpfo[1][2:end]
# _vov = getindex.(_vov, 1)

pf_x1s = getindex.(msx, 1)

scatter!(pax, collect(1:T), pf_x1s)

pfig


# _vov = tpfo[1][2:end]
# _vov = getindex.(_vov, 1)


Random.seed!(11)
# θ_in = rand(2)/2
# θ_in = [0.5, 0.2, 0.3]
θ_in = map(rand, (Float64, Float64, Float64))
tδ = 2e-5

keca = product_distribution(Beta(3,3), Beta(1.2, 1.2), Normal(0., 2.))

pθσ = 0.5 * collect(I(3))
pθ = MvNormal([0., 0., 0.], pθσ)
pθ = keca
#
# ll_hess_fwd(θ, test_pf)
# en_hess_fwd(θ, pθ, test_pf)

llv = ll_grad_zyg(θ_in, test_pf)
_nit = 400
θ_tp = Vector{Tuple{Float64, Float64, Float64}}(undef, _nit)
θ_tp[1] = θ_in
_grd_tp = Vector{Tuple{Float64, Float64, Float64}}(undef, _nit)
_grd_tp[1] = ll_grad_zyg(θ_in, test_pf)
_β = 0.2
_tsb = 75
_blhd = log_likelihood(test_pf, θ_in)
_islb = 1
_bindex = 1
for i = 2:_nit
    global _blhd
    if _islb > _tsb
        @info "Reverting to last best, lowering LR"
        θ_tp[i-1] = θ_tp[_bindex]
        _grd_tp[i-1] = _grd_tp[_bindex]
        global _islb = 0
        global tδ *= 0.85
    end
    _grd = ll_grad_zyg(θ_tp[i-1], test_pf)
    global θ_in_new = θ_tp[i-1] .+ tδ .* ((1-_β) .* _grd .+ _β .* _grd_tp[i-1])
    _tlhd = log_likelihood(test_pf, θ_in_new)
    if _tlhd == -Inf
        @warn "Out of bounds, reducing LR"
        global tδ *= 0.75
        θ_in_new[1] = NaN
    end
    println(θ_in_new)
    if _blhd < _tlhd
        global _islb = 0
        global _blhd = _tlhd
        global _bindex = i
    else
        global _islb += 1
    end

    if !any(isnan, θ_in_new)
        θ_tp[i] = θ_in_new
        _grd_tp[i] = _grd
    else
        @warn "NaN detected, reverting to a previous iterate"
        if i > 6
            θ_tp[i] = θ_tp[:, i-4]
            _grd_tp[i] = _grd_tp[:, i-4]
        else
            θ_tp[i] = θ_tp[i-1]
            _grd_tp[i] = _grd_tp[i-1]
        end
    end
end

tpfin = test_pf(θ_tp[:, _bindex], _store = true)

msxo = [0. for i in 1:T]

for i in 1:K
    local xs = [tpfin[1][t][i] for t in 1:T]
    msxo .+= xs
end

msxo ./= K


pfo_x1s = getindex.(msxo, 1)

scatter!(pax, collect(1:T), pfo_x1s)

_trmse = sqrt(mean((p_x1s - pf_x1s).^2))
_ermse = sqrt(mean((p_x1s - pfo_x1s).^2))
_rmse_params = sqrt(mean(θ_tp[:, _bindex] - θ).^2)

pfig

# gfig = Figure()
# gax = Axis(gfig[1,1])
#
# _plot_θ = collect(-0.5:0.1:1)
# _plot_grads = [mean([ll_grad_fwd([th, 0.1], test_pf) for i in 1:20]) for th in _plot_θ]
#
# _pgs = getindex.(_plot_grads, 1)
#
# lines!(gax, _plot_θ, _pgs)
# # vlines!(gax, 0.9, color = :black)
# hlines!(gax, 0.0, color = :black)
# gfig
