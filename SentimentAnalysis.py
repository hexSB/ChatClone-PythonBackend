
# from transformers import AutoModel, AutoTokenizer
import torch
from huggingface_hub import login
from environment_variables import HUGGING_TOKEN


login(token=HUGGING_TOKEN)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
low_cpu_mem_usage = True # Use less memory on CPU. Can be False.




import logging
from transformers import logging as hf_logging

logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()

# Now load the model and tokenizer
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    "goten9004/ConversationalModelLlama",
    load_in_4bit = load_in_4bit,
    low_cpu_mem_usage = low_cpu_mem_usage,
)
tokenizer = AutoTokenizer.from_pretrained("goten9004/ConversationalModelLlama")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
def get_sentiment(input_text):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "What is the sentiment of this conversation? Please choose an answer from {negative/neutral/positive}.", # instruction
                input_text, # input
                "", # output
            )
        ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    tokenizer.batch_decode(outputs)
    decoded_outputs = tokenizer.batch_decode(outputs)

    return decoded_outputs[0]
# print(get_sentiment("{user1: how was your day / user2: Kinda bad / user1: why? / user2: My dog got injured}") )



