from vllm import LLM, SamplingParams
import torch
import os
import time

from datasets import load_dataset
from tqdm import tqdm


# Constants.
WORLD_SIZE = 8
MODEL_DIR = "/home/azureuser/model"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, n=1, max_tokens=1)

# Create an LLM.
llm = LLM(model=os.path.join(MODEL_DIR, MODEL_NAME),
          tensor_parallel_size=WORLD_SIZE,
          enforce_eager=True)
print("Opened model!")

# Sample prompts
#prompts = [
#    "Hello, my name is",
#    "The president of the United States is",
#    "The capital of France is",
#    "The future of AI is",
#]
# Generate texts from sample prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
#for i in range(WORLD_SIZE):
#    with torch.cuda.device(i):
#        torch.cuda.synchronize()
#
#start_time = time.time()
#output = llm.generate(
#    prompts, sampling_params
#)
#end_time = time.time()
#elapsed_time = end_time - start_time
#print("Elapsed time: {} seconds. Throughput = {} prompts / sec".format(elapsed_time, len(prompts) / elapsed_time))

# MMLU
#task_list = [ "high_school_european_history", "business_ethics", "clinical_knowledge", "medical_genetics", "high_school_us_history", "high_school_physics", "high_school_world_history", "virology", "high_school_microeconomics", "econometrics", "college_computer_science", "high_school_biology", "abstract_algebra", "professional_accounting", "philosophy", "professional_medicine", "nutrition", "global_facts", "machine_learning", "security_studies", "public_relations", "professional_psychology", "prehistory", "anatomy", "human_sexuality", "college_medicine", "high_school_government_and_politics", "college_chemistry", "logical_fallacies", "high_school_geography", "elementary_mathematics", "human_aging", "college_mathematics", "high_school_psychology", "formal_logic", "high_school_statistics", "international_law", "high_school_mathematics", "high_school_computer_science", "conceptual_physics", "miscellaneous", "high_school_chemistry", "marketing", "professional_law", "management", "college_physics", "jurisprudence", "world_religions", "sociology", "us_foreign_policy", "high_school_macroeconomics", "computer_security", "moral_scenarios", "moral_disputes", "electrical_engineering", "astronomy", "college_biology", ]
#
#for subject in task_list:
#    dataset = load_dataset("lukaemon/mmlu", subject, split="test")
#    print(f'Test subject {subject} with {len(dataset)} samples.')
#    prompts = []
#    for example in tqdm(dataset, desc="Processing dataset"):
#        prompts.append(example['input'])
#    for i in range(WORLD_SIZE):
#        with torch.cuda.device(i):
#            torch.cuda.synchronize()
#    start_time = time.time()
#    output = llm.generate(
#        prompts, sampling_params
#    )
#    end_time = time.time()
#    elapsed_time = end_time - start_time
#    print("Elapsed time: {} seconds. Throughput = {} prompts / sec".format(elapsed_time, len(prompts) / elapsed_time))

# HuggingFaceH4/mt_bench_prompts
#for category in {'humanities', 'extraction', 'coding', 'stem', 'writing', 'reasoning', 'math', 'roleplay'}:
#dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
#print(f'Loaded dataset with {len(dataset)} samples.')
#prompts = []
#for example in tqdm(dataset['prompt'], desc="Processing dataset"):
#    prompts.append(example)
#for i in range(WORLD_SIZE):
#    with torch.cuda.device(i):
#        torch.cuda.synchronize()
#start_time = time.time()
#output = llm.generate(
#    prompts, sampling_params
#)
#end_time = time.time()
#elapsed_time = end_time - start_time
#print("Elapsed time: {} seconds. Throughput = {} prompts / sec".format(elapsed_time, len(prompts) / elapsed_time))

# openai_humaneval
#dataset = load_dataset("openai_humaneval", split="test")
#print(f'Loaded dataset with {len(dataset)} samples.')
#prompts = []
#for example in tqdm(dataset['prompt'], desc="Processing dataset"):
#    prompts.append(example)
#for i in range(WORLD_SIZE):
#    with torch.cuda.device(i):
#        torch.cuda.synchronize()
#start_time = time.time()
#output = llm.generate(
#    prompts, sampling_params
#)
#end_time = time.time()
#elapsed_time = end_time - start_time
#print("Elapsed time: {} seconds. Throughput = {} prompts / sec".format(elapsed_time, len(prompts) / elapsed_time))

# TIGER-Lab/MathInstruct
dataset = load_dataset("TIGER-Lab/MathInstruct", split="train")
print(f'Loaded dataset with {len(dataset)} samples.')
prompts = []
for example in tqdm(dataset['instruction'], desc="Processing dataset"):
    prompts.append(example)
for i in range(WORLD_SIZE):
    with torch.cuda.device(i):
        torch.cuda.synchronize()
start_time = time.time()
output = llm.generate(
    prompts, sampling_params
)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {} seconds. Throughput = {} prompts / sec".format(elapsed_time, len(prompts) / elapsed_time))
