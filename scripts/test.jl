using DrWatson
using Random
using LinearAlgebra
using CairoMakie

include("../src/dpf_stochad.jl")

#==

==#

Random.seed!(12)

prior(θ) = MvNormal(zeros(2), 0.1 .* collect(I(2)))

f_dyn(x, θ) = θ[1] .* x .+ (θ[2] .* x[1] .* x[2])

g_obs(x, θ) = x

state_model(x, θ) = MvNormal(f_dyn(x, θ), 0.1 .* collect(I(length(x))))
obs_model(x, θ) = MvNormal(g_obs(x, θ), 0.1 .* collect(I(length(x))))

Zygote.gradient(θ -> mean(rand(state_model([0.1, 0.5], θ))), [0.5, 0.1])

prop_model(x, y, θ) = MvNormal(f_dyn(x, θ), collect(I(length(x)))) # bootstrap pf

T = 50

θ = [0.9, 0.1]

test_SSM = StateSpaceModel(T, prior, state_model, obs_model, prop_model)

test_xs, test_ys = generate_fake_trajectory(test_SSM, θ)

p_x1s = getindex.(test_xs, 1)

pfig = Figure()
pax = Axis(pfig[1, 1])

scatter!(pax, collect(1:T), p_x1s)

pfig

K = 50

test_pf = ParticleFilter(K, test_SSM, test_ys, strat_sample)

tpfo = test_pf(θ, _store = true, _bpf = false)
#
# Random.seed!(1)
# _tgzg = ll_grad(θ, test_pf, _bpf = false)
# Random.seed!(1)
# _tgfw = ll_grad_fwd(θ, test_pf, _bpf = false)

_tgzg ≈ _tgfw

msx = [zeros(2) for i in 1:T]

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


θ_in = rand(2)/2
tδ = 5e-4

ll_grad(θ_in, test_pf, _bpf = false)
_nit = 400
θ_tp = zeros(2, _nit)
θ_tp[:, 1] = θ_in
_grd_tp = zeros(2, _nit)
_grd_tp[:, 1] = ll_grad_fwd(θ_in, test_pf, _bpf = false)
_β = 0.1
_tsb = 50
_blhd = log_likelihood(test_pf, θ_in)
_islb = 1
_bindex = 1
for i = 2:_nit
    global _blhd
    if _islb > _tsb
        @info "Reverting to last best, lowering LR"
        θ_tp[:, i-1] = θ_tp[:, _bindex]
        _grd_tp[:, i-1] .= _grd_tp[:, _bindex]
        global _islb = 0
        global tδ *= 0.5
    end
    _grd = ll_grad_fwd(θ_tp[:, i-1], test_pf, _bpf = false)
    global θ_in_new = θ_tp[:, i-1] + tδ * ((1-_β) .* _grd .+ _β .* _grd_tp[:, i-1])
    _tlhd = log_likelihood(test_pf, θ_in_new)
    println(θ_in_new)
    if _blhd < _tlhd
        global _islb = 0
        global _blhd = _tlhd
        global _bindex = i
    else
        global _islb += 1
    end

    if !any(isnan, θ_in_new)
        θ_tp[:, i] = θ_in_new
        _grd_tp[:, i] .= _grd
    else
        θ_tp[:, i] = θ_tp[:, i-1]
        _grd_tp[:, i] .= _grd_tp[:, i-1]
    end
end

tpfin = test_pf(θ_tp[:, _bindex], _store = true, _bpf = false)

msxo = [zeros(2) for i in 1:T]

for i in 1:K
    local xs = [tpfin[1][t][i] for t in 1:T]
    msxo .+= xs
end

msxo ./= K


pfo_x1s = getindex.(msxo, 1)

scatter!(pax, collect(1:T), pfo_x1s)

pfig

_trmse = sqrt(mean((p_x1s - pf_x1s).^2))
_ermse = sqrt(mean((p_x1s - pfo_x1s).^2))

gfig = Figure()
gax = Axis(gfig[1,1])

_plot_θ = collect(-1:0.05:1)
_plot_grads = [mean([ll_grad_fwd([th, 0.1], test_pf, _bpf = false) for i in 1:10]) for th in _plot_θ]

_pgs = getindex.(_plot_grads, 1)

lines!(gax, _plot_θ, _pgs)
vlines!(gax, 0.9, color = :black)
gfig
