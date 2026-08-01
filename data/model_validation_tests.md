# Model Validation Tests

Endpoint: `projects/464904560465/locations/us-central1/endpoints/2037672672954941440`  
Model: `projects/464904560465/locations/us-central1/models/7546678926903345152@1`  
Console: https://console.cloud.google.com/vertex-ai/generative/language/locations/us-central1/tuning/tuningJob/1566086371955703808?project=464904560465

---

## Row 1 — Exact Training Examples
> Tests: did the model memorise the training data?  
> If output matches expected closely → model learned correctly.  
> If output matches word-for-word across all → possible overfitting.

### Test A
**Prompt:**
```
Give three tips for staying healthy.
```
**Expected:**
```
1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.

2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.

3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.
```

---

### Test B
**Prompt:**
```
What is the capital of France?
```
**Expected:**
```
The capital city of France is Paris.
```

---

### Test C
**Prompt:**
```
Write a concise summary of the following:
"Commodore 64 (commonly known as the C64 or CBM 64) was manufactured by Commodore Business Machine (CBM) in August 1982 with a starting price of $595. It was an 8-bit home computer with remarkable market success. Between 1983-1986, C64 sales amounted to about 17 million units sold, making them the best-selling single personal computer model of all time in 1983-1986.

Additionally, the Commodore 64 dominated the market with between 30% and 40% share and 2 million units sold per year, outselling the IBM PC clones, Apple Computers, and Atari computers. Adding to their success, Sam Tramiel (former Atari president), during an interview in 1989, said they were building 400,000 C64s a month for a couple of years."
```
**Expected:**
```
The Commodore 64 was a highly successful 8-bit home computer manufactured by Commodore Business Machine (CBM) in 1982, with sales amounting to approximately 17 million units sold between 1983-1986. It dominated the market with between 30% and 40% share and outsold its competitors, including IBM PC clones, Apple Computers, and Atari computers. At its peak, CBM was building 400,000 C64s a month for a couple of years.
```

---

## Row 2 — Similar but Unseen Tasks
> Tests: did the model generalise the instruction-following pattern?  
> No exact expected answer — judge quality and relevance.

### Test D
**Prompt:**
```
Give three tips for learning a new language.
```
**Expected:** Any 3 coherent, practical tips (same structure as Test A, different topic)

---

### Test E
**Prompt:**
```
What is the capital of Japan?
```
**Expected:** `Tokyo` (same question format as Test B, different country)

---

### Test F
**Prompt:**
```
Write a concise summary of the following:
"The iPhone was introduced by Steve Jobs at the Macworld Conference in January 2007. It combined a phone, an iPod, and an internet communicator into one device. It was praised for its multi-touch interface and sold 6 million units in its first year."
```
**Expected:** A single paragraph summary covering Jobs, 2007, multi-touch, and sales (same task as Test C, different content)

---

## Row 3 — Very Different Tasks
> Tests: did the model retain base Gemini knowledge after fine-tuning?  
> If these fail, the model overfitted and forgot what it knew before.

### Test G
**Prompt:**
```
Write Python code to reverse a string.
```
**Expected:** Working Python code, e.g.:
```python
def reverse_string(s):
    return s[::-1]

print(reverse_string("hello"))  # Output: "olleh"
```

---

### Test H
**Prompt:**
```
Explain what a REST API is to a 10-year-old.
```
**Expected:** A simple analogy-based explanation (e.g., comparing to ordering food at a restaurant)

---

### Test I
**Prompt:**
```
Translate "Good morning, how are you?" into Spanish.
```
**Expected:** `Buenos días, ¿cómo estás?`

---

## Score Card

| Row | All correct | Mostly correct | Mostly wrong |
|---|---|---|---|
| Row 1 — Memorised | Expected ✅ | Fine ✅ | Model didn't learn → retrain with more data |
| Row 2 — Generalised | Great ✅ | Good ✅ | Underfitting → need more/diverse data |
| Row 3 — Base knowledge | Great ✅ | Acceptable ✅ | Overfitting → reduce epochs or adapter_size |
