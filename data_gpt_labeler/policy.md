# Google Reviews Trustworthiness Policy


## Background
You are given a CSV file containing Google reviews.
The columns are: ‘_id’, ‘rating’, ‘text’, ‘business_name’, ‘business_category’, and ‘business_description’.


## Task
Your task is to analyze each review (each row of the csv file) and add an extra attribute called "output".


- **output: 1** → Review is trustworthy.
- **output: 0** → Review is not trustworthy.


## Evaluation Guidelines
A review should be marked as trustworthy (output: 1) if it shows clear, specific, informative, and relevant evidence of being genuine.
If there is reasonable doubt about its authenticity or credibility, err on the side of marking it as not trustworthy (output: 0).

The following issues strongly reduce trustworthiness:


1. **Content mismatch** – The review's ‘text’ does not align with the ‘business_category’, ‘business_name’, or ‘business_description’.
2. **Overly generic** – The review ‘text’ lacks meaningful detail or substance.
3. **Uninterpretable** – The review ‘text’ has no interpretable meaning (e.g., random characters).
4. **Sentiment vs. Rating mismatch** – The ‘rating’ score contradicts the sentiment expressed in the ‘text’.
5. **Suspicious intent** – The review ‘text’ appears to be spam, an advertisement, a competitor bashing, AI-generated filler, or otherwise manipulative.
6. **Profanity/hate speech** – The review ‘text’ contains excessive profanity or hate speech.


## Instructions
If the overall content of the review suggests low credibility based on the above policies, mark it as "output": 0.
If the review generally appears genuine and credible, mark it as "output": 1.
Return the CSV file where for each review, "output" attribute is appended. Do not remove or modify any other fields.
