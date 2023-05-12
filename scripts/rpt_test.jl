using LinearAlgebra
using Zygote
using Distributions
using ProtoStructs

function f(a, b)
    m = Normal(a,b)
    s = rand(m, 5)
    return mean(s)
end

g = gradient(f, 1, 1)

function f_reparam(a, b)
    m = Normal(0,1)
    s = a .+ sqrt(b) .* rand(m, 5)
    return sum(s)
end

g_reparam = gradient(f_reparam, 1, 1)

@proto struct function_containing_struct
    f::Function
    x::Vector{Float64}
    y::Integer
end

k = function_containing_struct(x -> sum(0.5 .* x .^ 2), [1., 2., 3., 4.], 432)

g_fcs = gradient(k.f, k.x)

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

tf_g = gradient(tf, cs, visf)

kf(x,y) = [x * y, y^2]

j_kf = jacobian(kf, 1, 6)

function talloc(x, y)
    N = length(x)
    rv = Zygote.Buffer(x, N)
    for n = 1:N-1
        rv[n] = x[n] + x[n+1]
    end
    rv[N] = y
    return mean(rv)
end

tf_g = withgradient(talloc, [1., 2., 3.], 1.0)
