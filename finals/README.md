# REINFORCE Small Language Model Credit Assignment

## Project Overview

This project implements the REINFORCE algorithm from scratch and applies it to a small autoregressive character-level language model. The goal of the assignment is to understand temporal credit assignment in reinforcement learning, especially when the reward is only observed at the end of a generated sequence.

The model generates sequences using a vocabulary of five tokens:

- `<bos>`
- `a`
- `b`
- `c`
- `<eos>`

The reward is terminal-only. A sampled trajectory receives reward `1.0` if the substring `"abc"` appears anywhere in the generated sequence, and `0.0` otherwise.

This setup makes the task useful for studying credit assignment because early token decisions, such as choosing `a` after `<bos>`, only receive feedback after the full sequence has been sampled.

## Objective

The main objectives of this project are:

1. Implement a memoryless character-level policy network.
2. Sample trajectories from the policy using `Categorical(logits=...)`.
3. Define a sparse terminal reward function.
4. Train the policy using REINFORCE without a baseline.
5. Train a fresh policy using REINFORCE with a running-mean baseline.
6. Compare the reward curves of both training methods.
7. Analyze learned conditional distributions.
8. Explain how delayed reward provides gradient signal to earlier actions.

## Model Architecture

The policy network is a memoryless character-level model. It predicts the next token based only on the previously generated token.

The architecture contains:

- An embedding layer:
  - `nn.Embedding(vocab_size, 8)`
- A linear output layer:
  - `nn.Linear(8, vocab_size)`

The forward pass returns raw logits instead of probabilities. These logits are passed directly into PyTorch's `Categorical` distribution for sampling.

## Vocabulary

The project uses the following vocabulary:

```python
vocab = ['<bos>', 'a', 'b', 'c', '<eos>']
