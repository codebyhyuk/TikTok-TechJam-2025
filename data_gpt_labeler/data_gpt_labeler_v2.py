import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv


MAX_WORKERS = 10
REQUEST_DELAY = 1.5

INPUT_FILE = "/Users/yumin/Documents/GitHub/TikTok-TechJam-2025/data_gpt_labeler/filtered_datasets/final_data_2_filtered.csv" # change number after underbar in [1,8]
OUTPUT_FILE = "/Users/yumin/Documents/GitHub/TikTok-TechJam-2025/data_gpt_labeler/filtered_datasets/final_data_labeled_2.csv" # change number after underbar correspondingly

POLICY_FILE = "/Users/yumin/Documents/GitHub/TikTok-TechJam-2025/data_gpt_labeler/policy.md"
OPENAI_API_KEY = "OPENAI_API_KEY_YM"    #  OPENAI_API_KEY_HJ, OPENAI_API_KEY_HS, OPENAI_API_KEY_YM  #



# configure OPENAI_API
load_dotenv()
api_key = os.getenv(OPENAI_API_KEY)
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
client = OpenAI(api_key=api_key)


# read final data and policy
df = pd.read_csv(INPUT_FILE)
with open(POLICY_FILE, "r", encoding="utf-8") as f:
    policy_text = f.read()


# prompt function
def make_prompt(row):
    row_text = "\n".join([f"{col}: {row[col]}" for col in row.index])
    return f"""
{policy_text}

Row data:
{row_text}

Respond only with 0 (not trustworthy) or 1 (trustworthy).
"""


# labeling function
def label_row(idx, row):
    prompt = make_prompt(row)
    try:
        time.sleep(REQUEST_DELAY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        if "1" in text:
            label = "1"
        else:
            label = "0"
        print(f"Row {idx} labeled: {label}", flush=True)
        return idx, label
    except Exception as e:
        print(f"Error at row {idx}: {e}", flush=True)
        return idx, None


# multithreading main
def main():
    import time
    start_time = time.time()

    max_workers = MAX_WORKERS
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(label_row, idx, row) for idx, row in df.iterrows()]
        for future in as_completed(futures):
            idx, label = future.result()
            df.at[idx, "policy_label"] = label

    df.to_csv(OUTPUT_FILE, index=False)
    end_time = time.time()
    print(f"Labeling complete! Elapsed time: {end_time - start_time:.2f} seconds")


# run
if __name__ == "__main__":
    main()