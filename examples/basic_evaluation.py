"""Basic example of running a RAG evaluation."""

from rageval import (
    Document,
    EvalDataset,
    EvalQuery,
    ExperimentConfig,
    RAGEvaluator,
)


def main():
    # Create sample documents
    documents = [
        Document(
            doc_id="doc1",
            text="""Python is a high-level, general-purpose programming language.
            Its design philosophy emphasizes code readability with the use of significant indentation.
            Python is dynamically typed and garbage-collected.
            It supports multiple programming paradigms, including structured, object-oriented and functional programming.""",
        ),
        Document(
            doc_id="doc2",
            text="""JavaScript is a programming language that is one of the core technologies of the World Wide Web.
            It is a high-level, often just-in-time compiled language.
            JavaScript has dynamic typing, prototype-based object-orientation, and first-class functions.""",
        ),
        Document(
            doc_id="doc3",
            text="""Rust is a multi-paradigm, general-purpose programming language that emphasizes performance,
            type safety, and concurrency. It enforces memory safety without a garbage collector.""",
        ),
    ]

    # Create evaluation queries with ground truth
    queries = [
        EvalQuery(
            query="What programming language emphasizes code readability?",
            ground_truth_answer="Python emphasizes code readability with significant indentation.",
            relevant_chunk_ids=["doc1_0"],
        ),
        EvalQuery(
            query="Which language is used for the World Wide Web?",
            ground_truth_answer="JavaScript is one of the core technologies of the World Wide Web.",
            relevant_chunk_ids=["doc2_0"],
        ),
        EvalQuery(
            query="What language enforces memory safety without garbage collection?",
            ground_truth_answer="Rust enforces memory safety without a garbage collector.",
            relevant_chunk_ids=["doc3_0"],
        ),
    ]

    # Create dataset
    dataset = EvalDataset(
        documents=documents,
        queries=queries,
        name="programming_languages",
    )

    # Configure the experiment
    config = ExperimentConfig(
        name="basic_evaluation_example",
        description="Basic evaluation of RAG on programming language docs",
    )
    config.chunking.strategy = "recursive"
    config.chunking.chunk_size = 256
    config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
    config.generation.model = "mock"  # Use mock generator for example
    config.retrieval.top_k = 3
    config.metrics.k_values = [1, 3]

    # Run evaluation
    evaluator = RAGEvaluator(config)
    report = evaluator.evaluate(dataset)

    # Print results
    print(f"\n{'='*60}")
    print(f"Experiment: {report.experiment_name}")
    print(f"{'='*60}")

    print("\nRetrieval Metrics:")
    for metric, value in report.retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nAnswer Metrics:")
    for metric, value in report.answer_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nPerformance Metrics:")
    print(f"  Avg Retrieval Latency: {report.performance_metrics['avg_retrieval_latency_ms']:.2f}ms")
    print(f"  Throughput: {report.performance_metrics['throughput_qps']:.2f} queries/sec")

    print("\nError Analysis:")
    for error_type, rate in report.error_analysis["error_rates"].items():
        print(f"  {error_type}: {rate*100:.1f}%")

    # Save report
    report.save("results/basic_evaluation_report.json")
    print(f"\nReport saved to results/basic_evaluation_report.json")


if __name__ == "__main__":
    main()
