# Experiment design

Each experiment should declare four layers explicitly.

## 1. Physical rules

What can actually happen in the world?

Example: money cannot go negative; stealing has a probability of failure; movement takes time.

## 2. Social rules

What does the simulated society consider acceptable?

Example: stealing is socially condemned but physically possible.

## 3. Agent beliefs and goals

What does each agent believe and optimize for?

Example: security, wealth, belonging, influence, fairness, survival.

## 4. Perturbations

What event changes the environment?

Example: scarcity, rumor, election, football match, sudden inequality, new law.

## Anti-script rule

Do not encode the desired story into the consequences. If we want to study whether a rumor destabilizes the town, we model information flow and trust; we do not write `if rumor: cause_chaos()`.

## Reproducibility

Every run should record:

- seed
- model name + quantization
- system prompts
- experiment config
- all structured actions
- all state-changing events
- summary metrics
