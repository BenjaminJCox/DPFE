using LinearAlgebra
using Zygote
using ForwardDiff
using Distributions
using ProtoStructs
using DistributionsAD
using BenchmarkTools
using FiniteDiff
using ReverseDiff
# using StaticArrays
using Enzyme

function f(a, b)
    m = Normal(a,b)
    s = rand(m, 5)
    return mean(s)
end

g = Zygote.gradient(f, 1, 1)

function f_reparam(a, b)
    m = Normal(0,1)
    s = a .+ sqrt(b) .* rand(m, 5)
    return sum(s)
end

g_reparam = Zygote.gradient(f_reparam, 1, 1)

@proto struct function_containing_struct
    f::Function
    x::Vector{Float64}
    y::Integer
end

k = function_containing_struct(x -> sum(0.5 .* x .^ 2), [1., 2., 3., 4.], 432)

g_fcs = Zygote.gradient(k.f, k.x)

@proto struct con_struct
    x::Vector{Float64}
    y::Integer
end

cs = con_struct([1., 2., 3., 4.], 4)
visf(x,y) = mean(y .* x .^ 2)

function tf(cs, f)
    fcs = function_containing_struct(f, cs.x, cs.y)
    return fcs.f(fcs.x,fcs.y)
end

tf(cs, visf)

tf_g = Zygote.gradient(tf, cs, visf)

kf(x,y) = [x * y, y^2]

j_kf = Zygote.jacobian(kf, 1, 6)

function talloc(x, y)
    N = length(x)
    rv = Zygote.Buffer(x, N)
    for n = 1:N-1
        rv[n] = x[n] + x[n+1]
    end
    rv[N] = y
    return mean(rv)
end

tf_g = Zygote.withgradient(talloc, [1., 2., 3.], 1.0)

tf_comp(x) = sum(x[1:50].^2 ./ sqrt.(x[51:100]))

txs, tys = randn(50), rand(50)

ForwardDiff.gradient(tf_comp, vcat(txs, tys))
Zygote.gradient(tf_comp, vcat(txs, tys))
Enzyme.gradient(Enzyme.Reverse, tf_comp, vcat(txs, tys))
rd1 = ReverseDiff.gradient(tf_comp, vcat(txs, tys))

rdtape = ReverseDiff.compile(ReverseDiff.GradientTape(tf_comp, vcat(txs, tys)))
# @btime ReverseDiff.gradient!(rd1, rdtape, vcat(txs, tys)) seconds = 1
# @btime ForwardDiff.gradient(tf_comp, vcat(txs, tys)) seconds = 1
# @btime Zygote.gradient(tf_comp, vcat(txs, tys)) seconds = 1
# @btime FiniteDiff.finite_difference_gradient(tf_comp, vcat(txs, tys)) seconds = 1
# @btime Enzyme.gradient(Reverse, tf_comp, vcat(txs, tys)) seconds = 1
# @btime Enzyme.gradient(Forward, tf_comp, vcat(txs, tys)) seconds = 1

npdf(mu, sig, x) = inv(sig * sqrt(2π)) * exp(-0.5 * ((x - mu)/sig)^2)

function testcles(x)
    N = length(x)
    X = collect(1:N) .* x ./ 4
    X2 = abs.(X .+ 0.1)
    X = X .+ X2
    # X = map((x, y) -> npdf(x, 1, y), X, X2)
    X = map((x, y) -> pdf(MvNormal([x, x], [1. 0.; 0. 1.]), [y, y]), X, X2)
    return sum(X)
end

testcles(collect(-2:3))

# enzyme fails to compile here

fwd = ForwardDiff.gradient(testcles, collect(-2:200))
zgd = Zygote.gradient(testcles, collect(-2:200))[1]
# fid = FiniteDiff.finite_difference_gradient(testcles, 1. .* collect(-2:200))
rvd = ReverseDiff.gradient(testcles, 1. .* collect(-2:200))
rvd_t = ReverseDiff.compile(ReverseDiff.GradientTape(testcles, 1. .* collect(-2:200)))

# @btime ReverseDiff.gradient!(rvd, rvd_t, collect(-2:200)) seconds = 1
# @btime ForwardDiff.gradient(testcles, collect(-2:200)) seconds = 1
# @btime Zygote.gradient(testcles, collect(-2:200)) seconds = 1
# @btime FiniteDiff.finite_difference_gradient(testcles, 1. .* collect(-2:200)) seconds = 1

tfn(ms) = mean(rand(MvNormal(ms[1], ms[2])))

Zygote.gradient(tfn, [[1., 1.], [2. 0.5; 0.5 1.]])

function test_if_store(X)
    y = map(x -> x .^ 2, X)
    X = y
    return sum(X)
end

function test_if_store_G(X)
    return 2. .* X
end

t_xv = randn(20)
fwd_ts = sum(abs.(ForwardDiff.gradient(test_if_store, t_xv) - test_if_store_G(t_xv)))
zyg_ts = sum(abs.(Zygote.gradient(test_if_store, t_xv)[1] - test_if_store_G(t_xv)))

function test_dense_layer_dict(θ, x)
    return norm(θ[:A] * x .+ θ[:b], 1)
end

function test_dense_layer_unpack(θ, x, _t)
    A = reshape(θ[begin:(_t[1]*_t[2])], _t[2], _t[1])
    b = θ[(end-_t[2]+1):end]
    return norm(A * x .+ b, 1)
end


t_d = 25
t_x = randn(t_d)
t_theta = Dict(:A => rand(5, t_d), :b => randn(5))
t_theta_pack = vcat(vec(t_theta[:A]), t_theta[:b])
reshape(t_theta_pack[begin:5*t_d], 5, 25) == t_theta[:A]
t_theta_pack[end-4:end] == t_theta[:b]

_tv = test_dense_layer_dict(t_theta, t_x)

_x = copy(t_theta_pack)
_dx = zero(t_theta_pack)

zyg_dict = Zygote.gradient(θ -> test_dense_layer_dict(θ, t_x), t_theta)
fwd_pack = ForwardDiff.gradient(θ -> test_dense_layer_unpack(θ, t_x, (25, 5)), t_theta_pack)
enz_pack = Enzyme.autodiff(Enzyme.Reverse, θ -> test_dense_layer_unpack(θ, t_x, (25, 5)), Active, Duplicated(_x, _dx))

# @btime Zygote.gradient(θ -> test_dense_layer_dict(θ, t_x), t_theta)
# @btime ForwardDiff.gradient(θ -> test_dense_layer_unpack(θ, t_x, (25, 5)), t_theta_pack)

function _mvnlpdf(x, μ, Σ)
    vals, vecs = eigen(Σ)
    logdet = sum(log.(vals))
    dim = length(vals)
    log2pi = log(2 * π)
    dev = x - μ
    rtvalsinv = sqrt.(inv.(vals))
    U = vecs * diagm(rtvalsinv)
    maha = dev' * U
    sqmaha = maha * maha'
    return -0.5 * (dim * log2pi + sqmaha + logdet)
end

function naive_mvnlpdf(x, μ, Σ)
    dim = length(μ)
    log2pi = log(2.0 * π)
    logd = logdet(Σ)
    dev = x - μ
    sqma = dev' * inv(Σ) * dev
    return -0.5 * (dim * log2pi + sqma + logd)
end


x = rand(2)
μ = rand(2)
Σ = [1. -0.2; -0.2 1.3]

naive_mvnlpdf(x, μ, Σ) - logpdf(MvNormal(μ, Σ), x)

function test_logpdf(X)
    _l = 0.0
    z = zero(X[1, :])
    C = collect(1.0 .* I(length(z)))
    for i in 1:size(X, 1)
        # _l = _l + naive_mvnlpdf(z, X[i, :], C)
         _l = _l + logpdf(MvNormal(X[i, :], C), z)
    end
    return _l
end

ntx = 2.0 .* randn(20, 4)
dntx = zero(ntx)

tv = test_logpdf(ntx)

zyg_lmf = Zygote.gradient(θ -> test_logpdf(θ), ntx)[1]
fwd_lmf = ForwardDiff.gradient(θ -> test_logpdf(θ), ntx)
# enz_lmf = Enzyme.gradient(Enzyme.Reverse, θ -> test_logpdf(θ), ntx)
# enz_lmf = Enzyme.autodiff(Enzyme.Reverse, test_logpdf, Active, Duplicated(ntx, dntx))
# rvd_lmf_t = ReverseDiff.compile(ReverseDiff.GradientTape(test_logpdf, ntx))
# rvd_lmf = zero(ntx)
# ReverseDiff.gradient!(rvd_lmf, rvd_lmf_t, ntx)
# ReverseDiff.gradient(test_logpdf, ntx)


try
    enz_lmf = Enzyme.gradient(Enzyme.Reverse, θ -> test_logpdf(θ), ntx)
catch e
    io = IOBuffer()
    showerror(io, e)
    # global emsg = sprint((io,v) -> show(io, "text/plain", v), stacktrace(catch_backtrace()))
    global emsg = String(take!(io))
    write("error.txt", emsg)
end
# enz_lmf = Enzyme.autodiff(Enzyme.Reverse, θ -> test_logpdf(θ), Active, Active(ntx))
