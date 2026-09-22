# Running local agents with Ollama

Artificial Chaos Lab can use a local Ollama model as the decision policy for every simulated agent.

The project does **not** run one model per agent. One model stays loaded in Ollama and the simulation sends a new context for whichever agent is acting.

## Recommended first model

The first tested target is:

```text
qwen3:14b
```

For normal turns the API explicitly sends:

```json
"think": false
```

The model also receives a JSON Schema so actions are constrained to the simulation's action vocabulary.

## First run

Make sure Ollama is running and the model exists:

```powershell
ollama list
ollama ps
```

Then clone the repo and run a tiny experiment first:

```powershell
python -m src.acl.cli --policy ollama --model qwen3:14b --turns 3
```

Five agents acting for three turns means roughly fifteen model calls.

The objective event log is written to:

```text
logs/latest.jsonl
```

Each action event may include the model's short `reason`, which is useful for later analysis and replay.

## Information boundary

An agent receives:

- its own money, hunger, energy, food and reputation,
- names of other living agents,
- its own relationship values,
- its own memories,
- its own beliefs,
- public mechanics.

It does **not** receive the private money, hunger, energy, food, memories or beliefs of other agents.

This boundary is intentional. The world knows the ground truth; agents do not.

## Known mechanics vs hidden mechanics

The prompt explains the consequences an ordinary inhabitant could reasonably know.

Some implementation details can remain hidden. For example, the exact probability of a successful theft is enforced by the world but is not shown to the model.

This allows an agent's subjective estimate of risk to diverge from reality later.

## Useful CLI options

```text
--policy random|ollama
--model qwen3:14b
--ollama-url http://127.0.0.1:11434
--temperature 0.8
--context 4096
--turns 3
--seed 7
```

Start small. Once behavior looks correct, increase the number of turns.
