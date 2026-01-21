# RAGEvalLab

A configurable evaluation and benchmarking framework for Retrieval-Augmented Generation (RAG) systems.

## Overview

RAGEvalLab is designed to measure retrieval quality, answer accuracy, and system-level performance under controlled experiments. The framework supports pluggable components for systematic A/B comparisons across pipeline designs.

## Features

- **Pluggable Components**: Modular architecture for document chunking, embedding models, vector indexing (FAISS), reranking, and generation
- **Retrieval Metrics**: Precision@k, Recall@k, and Mean Reciprocal Rank (MRR)
- **Answer Quality Metrics**: Exact-match and F1 scoring
- **Performance Tracking**: End-to-end latency and throughput measurement
- **Error Analysis**: Automated classification of retrieval misses, incorrect chunk selection, and hallucinations
- **Reproducible Experiments**: Config-driven experiment runs with aggregated reports

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from rageval import RAGEvaluator
from rageval.config import ExperimentConfig

# Load configuration
config = ExperimentConfig.from_yaml("experiments/config.yaml")

# Run evaluation
evaluator = RAGEvaluator(config)
results = evaluator.run()

# Generate report
results.to_report("results/experiment_report.json")
```

## Project Structure

```
rageval/
├── chunking/       # Document chunking strategies
├── embedding/      # Embedding model interfaces
├── indexing/       # Vector index implementations (FAISS)
├── reranking/      # Reranking models
├── generation/     # LLM generation interfaces
├── metrics/        # Evaluation metrics
├── analysis/       # Error analysis tools
└── config/         # Configuration management
```

## Configuration

Experiments are defined via YAML configuration files:

```yaml
experiment:
  name: "baseline_evaluation"

chunking:
  strategy: "recursive"
  chunk_size: 512
  overlap: 50

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"

indexing:
  type: "faiss"
  index_type: "IVFFlat"

retrieval:
  top_k: 10

metrics:
  retrieval: ["precision@k", "recall@k", "mrr"]
  answer: ["exact_match", "f1"]
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
