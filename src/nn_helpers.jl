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
    return zeros(num_parameters)
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
    parameter_vector = generate_empty_pack(pack_tuple, has_bias)
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
