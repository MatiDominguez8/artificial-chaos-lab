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


## Portability rule

The core agent psychology must not assume a particular kind of world.

Generic traits such as risk tolerance, sociability, generosity, impulsivity,
competitiveness, trustfulness, curiosity and rule respect can travel between
experiments.

Scenario concepts do not belong in the core profile. Money, hunger, oxygen,
teams, votes, weapons, magic, jobs, laws, points or any other experiment-specific
resource/rule belong to the experiment.

Goals may be assigned per experiment because the same personality can pursue
different objectives in different situations.

Example:

```text
same generic agent profile
        │
        ├── Tiny Town       → accumulate resources
        ├── Space station   → keep the reactor alive
        ├── Reality show    → avoid elimination
        └── Absurd cult     → convince others the moon is fake
```

The reusable part is how the agent tends to think and react, not what world it
happens to inhabit.
