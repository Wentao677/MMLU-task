from datasets import load_dataset
import pandas as pd
import os

save_dir = "./mmlu_data"
os.makedirs(save_dir, exist_ok=True)

splits = [
    ("auxiliary_train", "auxiliary_train"),  
    ("dev", "validation"),        
    ("test", "test"),              
    ("validation", "validation")   
]

for save_name, hf_name in splits:
    try:
        print(f"\n {save_name} is downloading...")
        
        dataset = load_dataset(
            "cais/mmlu", 
            name="all",          
            split=hf_name,
            trust_remote_code=True
        )
        
        df = pd.DataFrame(dataset)
        save_path = os.path.join(save_dir, f"mmlu_{save_name}.jsonl")
        df.to_json(save_path, orient='records', lines=True, force_ascii=False)
        
        print(f"save {save_name} to {save_path}")
        print(f"There are: {len(df):,} in total")
        print("Sample looks like:")
        print(df.iloc[0])

    except Exception as e:
        print(f"{save_name} is failed with error: {str(e)}")
