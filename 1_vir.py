from transformers import AutoTokenizer
import transformers
import torch
from langchain.prompts import PromptTemplate
import sys


model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


template = """
write another paragraph to continue the story that follows, keeping the style of the story's author, Virginia Woolf:

{text}
"""

prompt = PromptTemplate(input_variables=["text"], template=template)

with open(sys.argv[1], 'r') as poem:
    text = poem.read()

poem_prompt = prompt.format(text=text)

sequences = pipeline(
    poem_prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")