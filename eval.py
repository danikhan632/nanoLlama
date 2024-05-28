import os
import torch
from contextlib import nullcontext
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)

# Configuration
max_questions = 100
num_samples = 1
max_new_tokens = 50
temperature = 0.8
top_k = 40
seed = 1337
device = 'cuda'
dtype = 'float16'

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)
model.eval()
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Set random seeds
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Function to run generation
def generate_text(prompt, num_samples=num_samples, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k):
    start_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    generated_texts = []
    with torch.no_grad():
        with ctx:
            for _ in range(num_samples):
                y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                generated_text = tokenizer.decode(y[0].tolist())
                generated_texts.append(generated_text)
    return generated_texts

# Load MMLU dataset
dataset = load_dataset("cais/mmlu", 'all', trust_remote_code=True)

# Evaluate model on MMLU dataset
num_correct = 0
num_questions = 0
for q in dataset['test']:
    if num_questions >= max_questions:
        break
    question = q['question']
    choices = q['choices']
    correct_answer_index = q['answer']
    correct_answer = choices[correct_answer_index]

    prompt = f"Question: {question}\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{i}. {choice}\n"
    prompt += "Answer:"

    generated_texts = generate_text(prompt, num_samples=1)
    generated_answer = generated_texts[0]
    
    generated_answer = generated_answer[len(prompt):]
    print(f"Q: {question}")
    print(f"Choices: {choices}")
    printc(f"Generated Answer: {generated_answer}")
    printc(f"Correct Answer: {correct_answer}", "green")
    print('---------------')

    # Check if the correct answer is in the generated text
    if str(correct_answer_index) in generated_answer or correct_answer in generated_answer:
        num_correct += 1
    num_questions += 1

accuracy = num_correct / num_questions if num_questions > 0 else 0
print(f"MMLU Evaluation Accuracy: {accuracy:.2%}")

# Sample text generation
start_prompt = "Once upon a time"
generated_texts = generate_text(start_prompt)
for text in generated_texts:
    print(text)
    print('---------------')
