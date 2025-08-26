import os
import pandas as pd
import asyncio
import time
from openai import OpenAI
from dotenv import load_dotenv


# configure gpt api
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")
client = OpenAI(api_key=api_key)


# read csv into dataframe
df = pd.read_csv("final_data_sampled.csv")


# read policy.md
with open("policy.md", "r", encoding="utf-8") as f:
    policy_text = f.read()


# prompt generation function
def make_prompt(row):
    row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    return f"""
            {policy_text}

            Row data:
            {row_text}

            Respond only with 0 (not trustworthy) or 1 (trustworthy).
            """


# async api call
CONCURRENCY = 50
semaphore = asyncio.Semaphore(CONCURRENCY)

async def label_row(row, idx):
    prompt = make_prompt(row)
    async with semaphore:
        try:
            response = await client.chat.completions.acreate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            label = response.choices[0].message.content.strip()
            label = "1" if "1" in label else "0"
            return idx, label
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            return idx, None


# main
async def main():
    tasks = [label_row(row, idx) for idx, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    for idx, label in results:
        df.at[idx, "policy_label"] = label

    df.to_csv("data_labeled.csv", index=False)
    print("Labeling complete!")


# run
start_time = time.time()
asyncio.run(main())
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")
