# What is this about
Code to collect LLM data using GPT

# How to use
### Install python package `gpt`
```shell
python -m pip install .
```
or
```shell
python -m pip install -e .[dev]
```


### Prometheus
- Train
```shell
python -m gpt.prometheus.train --infile {new_feedback_collection.json} --outfile {output filename} --cachefile cache.jsonl --max-concurrency {how many OpenAI API requests to run concurrently}
```

- Test
```shell
python -m gpt.prometheus.test --path {folder containing the Prometheus evaluation files} --cachefile cache.jsonl --max-concurrency {how many OpenAI API requests to run concurrently}
```
- the translated files will have the name `{original_name}_translated.json`, and have JSON format regardless of the original file format (JSON or NDJSON).


### Domain identification
- Collect subfields of predefined expert fields
```shell
python -m gpt.domain_identification.fields -n {number of subfields per primary fields} -o {outfile (.jsonl)}
```

- Collect keywords from the subfields
```shell
python -m gpt.domain_identification.keywords -n {number of keywords per the secondary fields above} -i {the secondary fields file (.jsonl)} -o {outfile (.jsonl)}
```

- Collect questions using the keywords
