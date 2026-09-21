# Experiment 001 — Tiny Town

## Question

If five autonomous agents receive the same starting resources and a small set of cooperative and antisocial actions, what patterns emerge over repeated turns?

## Starting conditions

- 5 agents
- 100 coins each
- 1 stored food each
- same initial hunger, energy and reputation
- fixed market food price

## Available actions

- work
- rest
- buy food
- talk
- give money
- steal money

## Important rule

The policy chooses the intention. The engine validates and applies consequences.

A model can say "steal 10 coins from Eva", but it cannot directly change Eva's balance.

## Phase 1

Run with `RandomPolicy` to validate state transitions, logs and reproducibility.

## Phase 2

Replace the policy with a local LLM through Ollama. Give each agent a personality, private objective and limited memory.

## Phase 3

Add a spatial world and replay UI.
