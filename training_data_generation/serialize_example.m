function out = serialize_example(example)

    out = "{ ""features"": { ";

    out = out + """feature"": { ";

    features = fieldnames(example);

    for idx = 1:numel(features)
        out = out + """" + features{idx} + """: { ";

        value = example.(features{idx});

        if isinteger(value)
            out = out + """int64List""";
        else
            out = out + """floatList""";
        end

        out = out + ": { ""value"": ";

        is_scalar = isscalar(value);

        if is_scalar
            out = out + "[";
        end

        out = out + jsonencode(value);

        if is_scalar
            out = out + "]";
        end

        out = out + " } }";

        if idx ~= numel(features)
            out = out + ",";
        end
    end

    out = out + " } } }";
end