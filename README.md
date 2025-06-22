# OpenGenius: Transformer-based Language Model from Scratch

OpenGenius is a transformer-based large language model (LLM) built entirely from scratch using PyTorch and NumPy. Inspired by the groundbreaking paper [“Attention is All You Need”](https://arxiv.org/abs/1706.03762), this project aims to provide a hands-on implementation of transformer architectures and showcase how context-aware text generation works at a fundamental level.

## Features

- Full transformer architecture (encoder-decoder)
- Positional encoding and token embeddings
- Multi-head self-attention
- Feed-forward layers with layer normalization
- Top-k sampling and temperature-based decoding
- Configurable model depth, hidden size, and heads

## Architecture Overview

OpenGenius uses a standard encoder-decoder transformer structure:
- Token Embedding + Positional Encoding
- N-layer Transformer Blocks
  - Multi-Head Attention
  - Add & Norm
  - Feed Forward + Dropout
- Decoder with masked self-attention and encoder-decoder cross-attention
- Output projection for vocabulary distribution
