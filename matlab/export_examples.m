number_of_examples = 50000;

examples = generate_examples(number_of_examples);

examples_serialized = repelem("", number_of_examples);

for idx = 1:number_of_examples
    examples_serialized(idx) = serialize_example(examples(idx));
end

examples_serialized = join(examples_serialized, newline);

file = fopen('training_data.tfrec', 'w');
fprintf(file, examples_serialized);