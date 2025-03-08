import json
import re
import argparse
import logging
import math
import requests
from openai import OpenAI
 
# Call the llm deployed in local environment
def call_local_model(prompt, num_return_sequences=1, max_new_tokens=2048, temperature=1.0, top_p=1.0, model_name=None, api_url="http://0.0.0.0:8888/v1/chat/completions"):

    logging.debug(f"[call_local_model] prompt: {prompt}")
    logging.debug(f"[call_local_model] num_return_sequences={num_return_sequences}, max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, model_name={model_name}, api_url={api_url}")

    payload = {

        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a knowledgeable reasoning assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "n": num_return_sequences

    }

    logging.debug(f"[call_local_model] Payload sent to local model: {json.dumps(payload, ensure_ascii=False)}")

    try:
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        logging.debug(f"[call_local_model] Response status code: {response.status_code}")
        logging.debug(f"[call_local_model] Raw response: {response.text}")

        if response.status_code == 200:
            result = response.json()
            completions = []
            for choice in result.get("choices", []):
                text_out = choice["message"]["content"].strip()
                completions.append({"generated_text": text_out})
            logging.debug(f"[call_local_model] completions: {completions}")
            return completions
        else:
            logging.error(f"Local model call failed: {response.status_code} {response.text}")
            return [{"generated_text": ""}] * num_return_sequences 

    except Exception as e:

        logging.error(f"Exception in call_local_model: {e}")

        return [{"generated_text": ""}] * num_return_sequences

 

 
# Dataloader
def load_mmlu_dataset(path):
    logging.debug(f"[load_mmlu_dataset] Loading dataset from path: {path}")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                data.append(sample)
                logging.debug(f"[load_mmlu_dataset] Loaded sample (line {line_num}): {sample}")
            except json.JSONDecodeError as e:
                logging.warning(f"[load_mmlu_dataset] JSON decode error at line {line_num}: {e}")
                continue
    logging.debug(f"[load_mmlu_dataset] Total samples loaded: {len(data)}")
    return data

 

# Format the MMLU raw data into prompt text
def format_question(sample):

    logging.debug(f"[format_question] Raw sample: {sample}")
    question = sample.get('question', '')
    if 'A' in sample and 'B' in sample and 'C' in sample and 'D' in sample:
        question_text = (f"{question}\nA. {sample['A']}\nB. {sample['B']}\n"
                         f"C. {sample['C']}\nD. {sample['D']}")
    elif 'choices' in sample:
        opts = sample['choices']
        if isinstance(opts, list) and len(opts) >= 4:
            question_text = (f"{question}\nA. {opts[0]}\nB. {opts[1]}\n"
                             f"C. {opts[2]}\nD. {opts[3]}")
        else:
            question_text = question
    else:
        question_text = question
    logging.debug(f"[format_question] Formatted question_text: {question_text}")

    return question_text

 

 

# Generator: Assume the parameter "choice" as the correct answer, generate the step-by-step rationales.
def generate_reasoning_paths(question_text, choice, n_paths, model_name, api_url):

    logging.debug(f"[generate_reasoning_paths] question_text: {question_text[:60]}..., choice: {choice}, n_paths={n_paths}, model_name={model_name}, api_url={api_url}")
    prompt = (f"{question_text}\nAssume the answer is {choice}. "
              f"Provide a step-by-step explanation for why this answer is correct, "
              f"strictly maintain the output in format of this template: Question itself. Step 1: ... Step 2 ... (as many Steps as you need). The answer is {choice}."
              )
    logging.debug(f"[generate_reasoning_paths] Final prompt: {prompt}")
    outputs = call_local_model(prompt, num_return_sequences=n_paths, max_new_tokens=2048,temperature=1.0, top_p=1.0, model_name=model_name, api_url=api_url)
    paths = []
    for out in outputs:
        reasoning = out['generated_text'].strip()
        logging.info(f"[Generation] Choice {choice} reasoning:\n{reasoning}")
        paths.append(reasoning)
    return paths

 

 
# Disassemble the rationales chains into child chains
def break_into_steps(reasoning_text):
    logging.debug(f"[break_into_steps] Original reasoning_text: {reasoning_text}")
    steps = []
    step_markers = [m.start() for m in re.finditer(r'\bStep\s*\d+:', reasoning_text)]
    logging.debug(f"[break_into_steps] step_markers: {step_markers}")
    if step_markers:
        for i, idx in enumerate(step_markers):
            if i < len(step_markers) - 1:
                prefix = reasoning_text[:step_markers[i+1]].strip()
            else:
                prefix = reasoning_text.strip()
            steps.append(prefix)
    else:
        sentences = re.split(r'(?<=[.!?])\s+', reasoning_text)
        cum_text = ""
        for sent in sentences:
            if sent:
                cum_text += sent.strip() + " "
                steps.append(cum_text.strip())
    logging.debug(f"[break_into_steps] Steps extracted: {steps}")
    return steps

 


 # Completor: Complete the following steps according to the given prefix chain.
def generate_completions_for_prefix(prefix_text, n_completions, model_name, api_url):
    logging.debug(f"[generate_completions_for_prefix] prefix_text: {prefix_text[:60]}..., n_completions={n_completions}")
    prefix_prompt = prefix_text.rstrip()

    prompt = (f"{prefix_prompt}\n"

              f"Above is the question and my rationales. Complete the remaining Steps,"

              f"you should strictly maintain the output in format of this template: Question itself. My rationales. Your following Steps(Same format with a 'Step number' in the start). The answer is (Choose the answer in its capital letter)."
           
              f"example: the question and my rationales: Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n\nStep 1: Simplify the radical expressions in the field extension.\n\nsqrt(3) is already in simplest form, so it remains the same.\nsqrt(18) can be simplified to sqrt(2) * sqrt(9), and since sqrt(9) = 3, sqrt(18) = sqrt(2) * 3.\nSo, the field extension is now Q(sqrt(2), sqrt(3), sqrt(2) * 3).\n\nStep 2: Find the minimal polynomial for sqrt(3) over Q. Since sqrt(3) is irrational, its minimal polynomial is x^2 - 3."

              f"In this case, you should output in this template: Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n\nStep 1: Simplify the radical expressions in the field extension.\n\nsqrt(3) is already in simplest form, so it remains the same.\nsqrt(18) can be simplified to sqrt(2) * sqrt(9), and since sqrt(9) = 3, sqrt(18) = sqrt(2) * 3.\nSo, the field extension is now Q(sqrt(2), sqrt(3), sqrt(2) * 3).\n\nStep 2: Find the minimal polynomial for sqrt(3) over Q. Since sqrt(3) is irrational, its minimal polynomial is x^2 - 3.\n\nStep 3: Find the minimal polynomial for sqrt(2) over Q. Since sqrt(2) is irrational, its minimal polynomial is x^2 - 2.\n\nStep 4: Find the minimal polynomial for sqrt(2) * 3 over Q. Since sqrt(2) * 3 can be rewritten as sqrt(2) * sqrt(9), and sqrt(9) = 3, sqrt(2) * 3 = sqrt(2) * 3. The minimal polynomial for sqrt(2) * 3 is the same as the minimal polynomial for sqrt(2), which is x^2 - 2.\n\nStep 5: Calculate the degree of the field extension Q(sqrt(2), sqrt(3), sqrt(2) * 3) over Q.\n\nThe minimal polynomial for sqrt(3) is x^2 - 3, which has degree 2.\nThe minimal polynomial for sqrt(2) is x^2 - 2, which has degree 2.\nThe minimal polynomial for sqrt(2) * 3 is x^2 - 2, which has degree 2. Since the three minimal polynomials are independent and have no common factors, the degree of the field extension is the product of their degrees, which is 2 * 2 * 1 = 4.\n\nAnswer: The answer is D, which is 6."
            )
    if not prompt.endswith('\n'):
        prompt += "\n"

    if "The answer is" in prefix_prompt:
        logging.debug("[generate_completions_for_prefix] Found final answer in prefix; not generating further completions.")
        return []  

    outputs = call_local_model(prompt, num_return_sequences=n_completions, max_new_tokens=2048, temperature=1.0, top_p=1.0, model_name=model_name, api_url=api_url)
    completions = []
    for out in outputs:
        text = out['generated_text']
        logging.debug(f"[generate_completions_for_prefix] Raw completion from model: {text}")
        if text.startswith(prefix_text):
            continuation = text[len(prefix_text):]
        else:
            continuation = text
        completion = continuation.strip()
        completions.append(completion)
        logging.info(f"[Expansion] From prefix '{prefix_text[:30]}...' -> continuation: {completion[:60]}...")

    logging.debug(f"[generate_completions_for_prefix] All completions: {completions}")

    return completions

 

 
#Check if the answer is correspond to the correct answer.
def check_answer(solution_text, correct_choice):
    logging.debug(f"[check_answer] solution_text: {solution_text}, correct_choice={correct_choice}")
    match = re.search(r"[Tt]he answer is\s+([A-D])", solution_text)
    if match:
        predicted = match.group(1).upper()
    else:
        match2 = re.search(r"\b([A-D])\b\.?$", solution_text.strip())
        predicted = match2.group(1).upper() if match2 else None
    logging.debug(f"[check_answer] Predicted answer: {predicted}")
    return 1 if predicted == str(correct_choice).upper() else 0

 

 
#Use fomula in ER-PRM to calculate the score for each step
def compute_step_scores(instance, correct_choice, entropy_coeff=2.0):
    logging.debug(f"[compute_step_scores] instance={instance}, correct_choice={correct_choice}, entropy_coeff={entropy_coeff}")
    scores = []

    for step_index, step in enumerate(instance[:-1]):
        completions = step.get('completions', [])
        logging.debug(f"[compute_step_scores] Step {step_index} prompt: {step['prompt']}")
        logging.debug(f"[compute_step_scores] completions: {completions}")
        labels = []

        for comp in completions:
            full_solution = step['prompt'] + " " + comp
            label_val = check_answer(full_solution, correct_choice)
            labels.append(label_val)

        if labels:
            p_correct = labels.count(1) / len(labels)
            p_wrong = labels.count(0) / len(labels)
            sum_exp = p_wrong * math.exp(entropy_coeff * 0) + p_correct * math.exp(entropy_coeff * 1)
            reward = (1.0 / entropy_coeff) * math.log(sum_exp + 1e-12)
            scores.append(round(abs(reward), 3))
            logging.debug(f"[compute_step_scores] Step {step_index} -> labels={labels}, p_correct={p_correct}, p_wrong={p_wrong}, sum_exp={sum_exp}, reward={reward}")

        else:
            scores.append(0.0)
            logging.debug(f"[compute_step_scores] Step {step_index} has no completions; appending 0.0")

    final_solution = instance[-1]['prompt']
    final_score = float(check_answer(final_solution, correct_choice))
    scores.append(round(final_score, 3))
    logging.debug(f"[compute_step_scores] Final step solution: {final_solution}, final_score={final_score}")
    return scores

 

 
# Assemble all steps seperated by 'ки'
def combine_steps_with_placeholders(step_texts, scores):
    logging.debug(f"[combine_steps_with_placeholders] step_texts count: {len(step_texts)}, scores: {scores}")

    if not step_texts:
        return "", []
    combined_text = step_texts[0].strip()

    for i in range(1, len(step_texts)):
        prev = step_texts[i - 1]
        curr = step_texts[i]
        # If the current step starts with the entire previous step, just take the 'diff'
        diff = curr[len(prev):] if curr.startswith(prev) else curr
        combined_text += " ки\n" + diff.strip()

    logging.debug(f"[combine_steps_with_placeholders] combined_text: {combined_text[:200]}...")  
    return combined_text.strip(), scores

 

 
# Main function
def optimize_mmlu(dataset_path, n_paths, n_reasonings, sample_only, output_path, model_name, entropy_coeff, api_url):

    logging.info(f"Loading dataset from {dataset_path}")
    data = load_mmlu_dataset(dataset_path)

# Quick Validation
    if sample_only:
        data = data[:2] 
        logging.info("[optimize_mmlu] sample_only set, taking only first 2 items.")

    logging.info(f"Total questions to process: {len(data)}")
    optimized_entries = []
    for idx, sample in enumerate(data, start=1):
        question_text = format_question(sample)
        correct = sample.get('answer') or sample.get('correct') or sample.get('label')
        logging.debug(f"[optimize_mmlu] sample index {idx}, raw 'correct' value: {correct}")

        if correct is None:
            logging.warning(f"Question {idx}: no correct answer provided, skipping.")
            continue

        correct_choice = str(correct).strip()

        if correct_choice.isdigit():
            mapping = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
            correct_choice = mapping.get(correct_choice, correct_choice)
        logging.info(f"Question {idx}: {question_text.splitlines()[0]}... (Correct: {correct_choice})")

        for choice in ['A', 'B', 'C', 'D']:
            reasoning_paths = generate_reasoning_paths(question_text, choice, n_reasonings, model_name, api_url)
            logging.debug(f"[optimize_mmlu] reasoning_paths for choice {choice}: {reasoning_paths}")

            for reasoning_index, reasoning in enumerate(reasoning_paths, start=1):
                logging.debug(f"[optimize_mmlu] reasoning path #{reasoning_index}, text:\n{reasoning}")
                prefixes = break_into_steps(reasoning)
                instance = []

                for j, prefix in enumerate(prefixes):           
                    if j < len(prefixes) - 1:
                        completions = generate_completions_for_prefix(prefix, n_paths, model_name, api_url)
                        instance.append({"prompt": prefix, "completions": completions})
                    else:
                        instance.append({"prompt": prefix})
                step_scores = compute_step_scores(instance, correct_choice, entropy_coeff)
                logging.debug(f"[optimize_mmlu] step_scores: {step_scores}")
                step_texts = [step['prompt'] for step in instance]
                combined_text, values = combine_steps_with_placeholders(step_texts, step_scores)
                optimized_entry = {"text": combined_text, "value": values}
                optimized_entries.append(optimized_entry)
                logging.info(f"Finished reasoning path for choice {choice}, path #{reasoning_index}: scores = {values}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(optimized_entries, f_out, indent=4, ensure_ascii=False)
    logging.info(f"Optimization complete. Saved {len(optimized_entries)} entries to '{output_path}'.")

 

 







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Use ER-PRM to process MMLU in a MCTS way")
    parser.add_argument("--dataset", type=str, default="/home/guest/test/mmlu_test.jsonl", help="MMLU JSONL Path")
    parser.add_argument("--n_paths", type=int, default=5, help="how many paths should a mid-step generate")
    parser.add_argument("--n_reasonings", type=int, default=5, help="how many paths should a root node(a choice) generate")
    parser.add_argument("--sample", action="store_true", help="For a quick validation, only process the first two data")
    parser.add_argument("--output_file", type=str, default="output/optimized_mmlu_dataset.json", help="Output path")
    parser.add_argument("--model", type=str, default="/home/guest/models/llama-8B-instruct", help="Local llm path, like '/home/guest/models/llama-8B-instruct'")
    parser.add_argument("--entropy_coeff", type=float, default=2.0, help="entropy regulation coefficient")
    parser.add_argument("--api_url", type=str, default="http://0.0.0.0:8888/v1/chat/completions", help="Local llm api")
    parser.add_argument("--log_file", type=str, default="log/mmlu_optimization.log", help="Logging path")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        level=logging.DEBUG, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("=== Start MMLU processing ===")
    optimize_mmlu(
        dataset_path=args.dataset,
        n_paths=args.n_paths,
        n_reasonings=args.n_reasonings,
        sample_only=args.sample,
        output_path=args.output_file,
        model_name=args.model,
        entropy_coeff=args.entropy_coeff,
        api_url=args.api_url
    )

    logging.info("=== Finished! ===")