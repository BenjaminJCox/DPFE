using Distributions
using DistributionsAD
using LinearAlgebra

function evaluate_pack(X, packed, pack_tuple, has_bias, activations)
    n_layers = length(pack_tuple) - 1
    layer_start_idx = 1
    current_value = copy(X)
    for layer in 1:n_layers
        input = pack_tuple[layer]
        output = pack_tuple[layer+1]
        matrix = reshape(packed[layer_start_idx:(layer_start_idx + input*output - 1)], output, input)
        current_value = matrix * current_value
        if has_bias[layer] == true
            # @info length(current_value)
            # @info length(packed[(layer_start_idx + input*output):(layer_start_idx + input*output + output - 1)])
            current_value = current_value + packed[(layer_start_idx + input*output):(layer_start_idx + input*output + output - 1)]
            layer_start_idx = (layer_start_idx + input*output + output)
        else
            layer_start_idx = (layer_start_idx + input*output)
        end
        current_value = activations[layer](current_value)
    end
    return current_value
end

function generate_empty_pack(pack_tuple, has_bias)
    n_layers = length(pack_tuple) - 1
    num_parameters = 0
    for layer in 1:n_layers
        input = pack_tuple[layer]
        output = pack_tuple[layer+1]
        matrix_size = input * output
        if has_bias[layer] == true
            bias_size = output
        else
            bias_size = 0
        end
        num_parameters = num_parameters + matrix_size + bias_size
    end
    return (zeros(num_parameters), pack_tuple, has_bias)
end

function pack_known(matrices, biases)
    n_layers = length(matrices)
    has_bias = biases .!= zero.(biases)
    pack_vector = zeros(Int, n_layers+1)
    for _idx in 1:n_layers
        _sz = size(matrices[_idx])
        # input
        pack_vector[_idx] = _sz[2]
        # output
        pack_vector[_idx+1] = _sz[1]
    end
    pack_tuple = Tuple(pack_vector)
    display(pack_tuple)
    display(has_bias)
    parameter_vector = generate_empty_pack(pack_tuple, has_bias)[1]
    layer_start_idx = 1
    for layer in 1:n_layers
        input = pack_tuple[layer]
        output = pack_tuple[layer+1]
        # @info(length(matrices[layer]))
        # @info(length(parameter_vector[layer_start_idx:(layer_start_idx + input*output - 1)]))
        parameter_vector[layer_start_idx:(layer_start_idx + input*output - 1)] = matrices[layer][:]
        if has_bias[layer]
            @info(length(biases[layer]))
            @info(length(parameter_vector[(layer_start_idx + input*output):(layer_start_idx + input*output + output-1)]))
            parameter_vector[(layer_start_idx + input*output):(layer_start_idx + input*output + output-1)] .= biases[layer]
            layer_start_idx = (layer_start_idx + input*output + output)
        else
            layer_start_idx = layer_start_idx + input*output
        end
    end
    return (parameter_vector, pack_tuple, has_bias)
end

function MVNM_pack(state_dim, obs_dim, num_comps, isotropic = true; layers = [128, 256], bias = [true, true, true, true])
    # @assert length(bias) == (length(layers) - 2)
    # @info(length(layers))
    # @info length(bias)
    input_dim = state_dim + obs_dim
    output_dim = num_comps * state_dim + num_comps
    pack_size = length(layers) + 2
    pack_vector = zeros(Int, pack_size)
    pack_vector[begin] = input_dim
    pack_vector[end] = output_dim
    pack_vector[begin+1:end-1] .= layers
    pack_tuple = Tuple(pack_vector)
    bias_tuple = Tuple(bias)
    return generate_empty_pack(pack_tuple, bias_tuple)
end

rl(x) = max.(x, 0.0)

function MVNM_unpack(x, y, pack, num_comps, activations, isotropic = true)
    _input = vcat(x, y)
    state_dim = length(x)
    n_fns = length(pack[2]) - 1
    # activations = Vector{Function}(undef, n_fns)
    # ignore_derivatives() do
    #     # activations .= [identity for fn in 1:n_fns]
    #     activations[begin:end-1] .= rl
    #     activations[end] = identity
    #     @info length(activations)
    # end
    _output = evaluate_pack(_input, pack..., activations)

    # means = []
    # cov_scales = []
    cov_idx_start = state_dim*num_comps
    # for comp in 1:num_comps
    #     crm = _output[(comp-1)*state_dim+1:comp*state_dim]
    #     push!(means, crm)
    #     csc = _output[cov_idx_start+comp]
    #     push!(cov_scales, csc)
    # end
    #
    # mm_vector = []
    # for comp in 1:num_comps
    #     push!(mm_vector, MvNormal(means[comp], cov_scales[comp] .* collect(I(state_dim))))
    # end
    mm_vector = [TuringDenseMvNormal(_output[(comp-1)*state_dim+1:comp*state_dim], _output[cov_idx_start+comp] .* collect(I(state_dim))) for comp in 1:num_comps]

    result_model = MixtureModel(TuringDenseMvNormal[mm_vector...])
    return result_model
end

function init_learnable_proposal(state_dim, obs_dim, num_comps, isotropic = true; layers = [128, 256], bias = [true, true, true, true], activations = [rl, rl, identity])
    _θ, pack_tuple, bias_tuple = MVNM_pack(state_dim, obs_dim, num_comps, isotropic; layers = layers, bias = bias)
    _prop(x, y, θ) = MVNM_unpack(x, y, (θ, pack_tuple, bias_tuple), num_comps, activations, isotropic)
    return _prop, _θ, pack_tuple
end
