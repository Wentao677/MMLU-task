import os
import json
import argparse
from openai import OpenAI

def rephrase_question(client, original_question: str) -> str:

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Please rephrase the following question in a way that preserves its meaning, "
                    "but do not alter the answer choices or the correct answer."
                )
            },
            {
                "role": "user",
                "content": original_question
            }
        ],
        temperature=0.7,
        max_tokens=256,
    )
    rephrased = response.choices[0].message.content.strip()
    return rephrased

def main():
    parser = argparse.ArgumentParser(description="Rephrase local MMLU dataset questions using DeepSeek API.")
    parser.add_argument(
        "--dataset",
        choices=["mmlu_auxiliary_train", "mmlu_dev", "mmlu_test", "mmlu_validation"],
        required=True,
        help="Assign MMLU JSONL file"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Only process the first 20 data for a quick validation"
    )
    args = parser.parse_args()
    data_dir = "mmlu_data"
    os.makedirs(data_dir, exist_ok=True) 
    dataset_file = os.path.join(data_dir, f"{args.dataset}.jsonl")
    if not os.path.exists(dataset_file):
        print(f"No such file {dataset_file}")
        return
    with open(dataset_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = [json.loads(line.strip()) for line in lines]
    if args.sample:
        data = data[:20]

    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com", 
    )

    rephrased_data = []
    for idx, item in enumerate(data):
        original_question = item["question"]
        try:
            new_question = rephrase_question(client, original_question)
        except Exception as e:
            print(f"[Warning] {idx} is wrong while processing, will remain the raw one {e}")
            new_question = original_question 

        new_item = {
            "question": new_question,
            "subject": item.get("subject", ""),
            "choices": item["choices"],
            "answer": item["answer"]
        }
        rephrased_data.append(new_item)


    output_file = os.path.join(data_dir, f"{args.dataset}_rephrased.jsonl")
    with open(output_file, "w", encoding="utf-8") as fw:
        for record in rephrased_data:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"The rephrased file is saved in: {output_file}")

if __name__ == "__main__":
    main()
