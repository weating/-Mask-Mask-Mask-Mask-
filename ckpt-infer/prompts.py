inference_prompts = """
# Mathematical Proof Derivation Restoration Task

## Task Description
Given a mathematical proof derivation process where key steps are masked with [MASK]_n tokens, please restore the masked mathematical expressions or conclusions based on proof logic, mathematical theorems, and derivation context.

## Analysis Requirements
1. **Proof Logic**: Understand the overall reasoning and objective of the proof
2. **Mathematical Rigor**: Ensure derivations comply with mathematical theorems and properties
3. **Symbol Consistency**: Maintain consistency with symbol definitions in the context
4. **Derivation Coherence**: Ensure natural logical connections between consecutive steps

## Input Proof
{proof_text}

## Restoration Steps
1. Identify the mathematical background of the proof (optimization theory, probability theory, analysis, etc.)
2. Analyze established equations and derivation chains
3. Understand the role of [MASK] positions in the proof (intermediate results, final conclusions, etc.)
4. Apply relevant mathematical properties (such as unbiasedness, expectation properties, norm properties, etc.)

## Output Format
**[MASK]_1 Restoration Result:**
$$LaTeX formatted mathematical expression$$

**Derivation Basis:**
- Mathematical properties or theorems used
- Logical relationship with preceding text
- Key step explanations for the derivation

**Verification:**
Brief verification of the reasonableness of the restoration result
"""