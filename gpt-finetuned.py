# -*- coding: utf-8 -*-
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer,GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-finetuned")
model = GPT2LMHeadModel.from_pretrained("gpt2-finetuned")

# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Path to your fine-tuned model
model_path ="gpt2-finetuned"

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_text(input_text, model, tokenizer, max_length=200):
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate output text
    output_ids = model.generate(input_ids, max_length=max_length,temperature=0.7,do_sample=True)

    # Decode the output text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

# Example usage
prompt="Give a generic information on the subject under 75 words."
skip_len=len(prompt)
input_text = "Surveillence Drone"
prompt+=input_text
output_text = generate_text(prompt, model, tokenizer)
print(output_text[skip_len:],flush=True)

            


