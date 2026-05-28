# Prompt Diff — Permissive V1 vs Strict V2

**Key change:** V1 says 'answer using context' (preference). V2 says 'answer ONLY IF directly relevant' + explicit refusal text + citation instruction ([Source: chunk_id]).

---

## Query 1: Direct in-scope

**Question:** What is Newton's second law of motion?

**Note:** Both should answer; V2 adds citations

### V1 Response (Permissive Prompt)

```
Newton's second law of motion states that the rate of change of momentum of an object is proportional to the applied unbalanced force in the direction of the force. Mathematically, it can be expressed as:

F = ma

Where:
- F is the force applied to the object
- m is the mass of the object
- a is the acceleration produced in the object

In other words, the second law of motion gives us a method to measure the force acting on an object as a product of its mass and acceleration. The unit of force is kg m s-2 or newton, which has the symbol N.
```

**V1 verdict:** Answered | Citations: none

### V2 Response (Strict Prompt)

```
Newton's second law of motion states that the rate of change of momentum of an object is proportional to the applied unbalanced force in the direction of the force [force_and_laws_of_motion_054].
```

**V2 verdict:** Answered | Citations: none

---

## Query 2: Out-of-scope (Biology)

**Question:** Explain how photosynthesis works in plants.

**Note:** V1 may hallucinate; V2 must refuse

### V1 Response (Permissive Prompt)

```
Unfortunately, the provided context does not have any information about photosynthesis. However, I can explain how photosynthesis works in plants.

Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy in the form of glucose. This process occurs in specialized organelles called chloroplasts, which are present in plant cells.

The overall equation for photosynthesis is:

6 CO2 + 6 H2O + light energy → C6H12O6 (glucose) + 6 O2

The process of photosynthesis involves two stages: the light-dependent reactions and the light-independent reactions.

**Light-dependent reactions:**

1. Light is absorbed by pigments such as chlorophyll and other accessory pigments in the thylakoid membrane of the chloroplast.
2. The energy from light is used to generate ATP and NADPH.
3. Water is split into oxygen, protons, and electrons.

**Light-independent reactions (Calvin cycle):**

1. CO2 is fixed into a three-carbon molecule called 3-phosphoglycerate (3-PGA) using the energy from ATP and NADPH.
2. The 3-PGA molecules are reduced to form glyceraldehyde-3-phosphate (G3P).
3. The G3P molecules are used to synthesize glucose.

In summary, photosynthesis is the process by which plants convert light energy into chemical energy in the form of glucose, releasing oxygen as a byproduct. This process is essential for life on Earth, as it provides energy and organic compounds for plants to grow and thrive.
```

**V1 verdict:** Answered | Citations: none

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
The provided context does not directly explain how electric current flows through a conductor. However, I can provide a general explanation.

Electric current flows through a conductor due to the movement of charged particles, such as electrons. When a conductor is connected to a power source, the electrons in the conductor gain energy and start moving. This movement of electrons is what we call electric current.

In the context of the provided text, the electric bell is mentioned, but it does not explain the flow of electric current through the conductor. However, I can provide a general explanation.

When an electric current flows through a conductor, it is due to the movement of electrons from one end of the conductor to the other. This movement of electrons is caused by the electric field created by the power source. The electrons in the conductor gain energy and start moving, creating an electric current.

If you would like to know more about electric current and its flow through conductors, I can provide more information.
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
