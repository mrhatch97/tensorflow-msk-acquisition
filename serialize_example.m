function out = serialize_example(example)
    out = "features {" + newline;

    indent = sprintf('\t');

    features = fieldnames(example);

    for idx = 1:numel(features)
        out = out + indent + "feature { ";

        out = out + "key: " + features{idx} + " ";
        out = out + "value { ";

        value = example.(features{idx});

        if isinteger(value)
            out = out + "int64_list ";
        else
            out = out + "float_list ";
        end

        is_scalar = isscalar(value);

        if is_scalar
            out = out + "[";
        end

        out = out + jsonencode(value);

        if is_scalar
            out = out + "]";
        end

        out = out + " } }" + newline;
    end

    out = out + "}";
end