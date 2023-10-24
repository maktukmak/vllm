from vllm import LLM, SamplingParams
import time

input_len = 64
batch_size = 32

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.8,
                                 top_p=0.95,
                                 ignore_eos=True,
                                 max_tokens = 32
                                 )


llm = LLM(model="facebook/opt-125m",
          tokenizer = "facebook/opt-125m",
          dtype='float32', cpu_only = True)



dummy_prompt_token_ids = [[0] * input_len] * batch_size
start = time.time()
outputs = llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                       sampling_params=sampling_params,
                       )
print('Latency batched (sec):', time.time() - start)


start = time.time()
for prompt in dummy_prompt_token_ids:
    outputs = llm.generate(prompt_token_ids=[prompt],
                       sampling_params=sampling_params,
                       )
print('Latency single (sec):', time.time() - start)



# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")