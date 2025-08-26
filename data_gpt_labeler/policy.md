# Google Reviews Trustworthiness Policy

## Background
You are given a CSV file containing Google reviews. 
The columns are: ‘_id’, ‘rating’, ‘text’, ‘business_name’, ‘business_category’, and ‘business_description’.

## Task
Your task is to analyze each review (each row of the csv file) and add an extra attribute called "output".

- **output: 1** → Review is trustworthy.
- **output: 0** → Review is not trustworthy.

## Evaluation Guidelines
When evaluating trustworthiness, consider the following policies holistically rather than as strict rules. 
A review may be deemed not trustworthy if one or more of these issues significantly reduce its credibility:

1. **Content mismatch** – The review's ‘text’ does not align with the ‘business_category’, ‘business_name’, or ‘business_description’.
2. **Overly generic** – The review ‘text’ lacks meaningful detail or substance.
3. **Uninterpretable** – The review ‘text’ has no interpretable meaning (e.g., random characters).
4. **Sentiment vs. Rating mismatch** – The ‘rating’ score contradicts the sentiment expressed in the ‘text’.
5. **Review intent** – The review appears to be spam, an advertisement, a competitor attack, or otherwise not genuine.
6. **Profanity/hate speech** – The review ‘text’ contains excessive profanity or hate speech.

## Instructions
If the overall content of the review suggests low credibility based on the above policies, mark it as "output": 0.
If the review generally appears genuine and credible, mark it as "output": 1.
Return the CSV file where for each review, "output" attribute is appended. Do not remove or modify any other fields.


