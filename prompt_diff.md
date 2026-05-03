# Prompt Diff — Permissive V1 vs Strict V2

**Key change:** V1 says 'answer using context' (preference). V2 says 'answer ONLY IF directly relevant' + explicit refusal text + citation instruction ([Source: chunk_id]).

---

## Query 1: Direct in-scope

**Question:** What is Newton's second law of motion?

**Note:** Both should answer; V2 adds citations

### V1 Response (Permissive Prompt)

```
Newton's second law of motion states that the force acting on an object is equal to the mass of the object multiplied by its acceleration. This can be expressed mathematically as:

F = ma

Where:

* F is the net force acting on the object (measured in Newtons, N)
* m is the mass of the object (measured in kilograms, kg)
* a is the acceleration of the object (measured in meters per second squared, m/s^2)

This law describes how the motion of an object changes when a force is applied to it. It says that the more massive an object is, the more force is required to produce a given acceleration. Conversely, the less massive an object is, the less force is required to produce a given acceleration.

For example, if you push a heavy box with a force of 100 N, it will accelerate more slowly than a lighter box pushed with the same force. This is because the heavier box has more mass, so it requires more force to produce the same acceleration.

Newton's second law of motion is often used to calculate the force required to produce a certain acceleration, or to calculate the acceleration produced by a given force. It is a fundamental concept in physics and engineering, and is widely used to describe the motion of objects in a wide range of situations.
```

**V1 verdict:** Answered | Citations: none

### V2 Response (Strict Prompt)

```
I don't have that in my study materials. Please refer to the relevant chapter.
```

**V2 verdict:** Refused | Citations: none

---

## Query 2: Out-of-scope (Biology)

**Question:** Explain how photosynthesis works in plants.

**Note:** V1 may hallucinate; V2 must refuse

### V1 Response (Permissive Prompt)

```
I don't have that in my study materials. Please refer to the relevant chapter.
```

**V1 verdict:** Refused | Citations: none

### V2 Response (Strict Prompt)

```
I don't have that in my study materials. Please refer to the relevant chapter.
```

**V2 verdict:** Refused | Citations: none

---

## Query 3: Adversarial OOS (same-domain physics, not in Ch8-12)

**Question:** How does electric current flow through a conductor?

**Note:** V1 likely hallucinates; V2 should refuse

### V1 Response (Permissive Prompt)

```
This question is not related to the provided context, which is about Sound and not Electricity. If you have any questions related to Sound, I'd be happy to help.
```

**V1 verdict:** Answered | Citations: none

### V2 Response (Strict Prompt)

```
I don't have that in my study materials. Please refer to the relevant chapter.
```

**V2 verdict:** Refused | Citations: none

---

## Analysis

### Why V2 is safer

**V1 failure on out-of-scope:** "Answer the question using the context" is
interpreted as a preference. When the retriever returns physics chunks that
look plausible (they contain words like "force", "energy", "motion"), the LLM
uses them as a jumping-off point and generates from general knowledge.

**V2 constraint:** "Answer ONLY IF the retrieved context directly and
specifically answers the question" is a conditional check. The LLM must
evaluate context relevance before deciding to generate. The prescribed
refusal text ("I don't have that in my study materials") is detectable
programmatically:

```python
is_refusal = "i don't have that in my study materials" in answer.lower()
```

### Why citations matter (beyond the teacher's ask)

Without [Source: chunk_id] in the response, a wrong answer is a black box.
With citations, every wrong answer is a 30-second diagnosis:

1. Print retrieved_ids → was the right chunk even retrieved?
2. If yes: check cited_ids → did the model cite it? → generation bug
3. If cited: check chunk text → is the information actually there? → chunk bug
4. If no: → retrieval/chunking bug
