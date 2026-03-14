import json
import subprocess
import time

METRICS_SCHEMA = [
    {"name": "accuracy", "type": "float", "goal": "maximize"},
    {"name": "latency_ms", "type": "float", "goal": "minimize"},
]

def evaluate():
    expected = "hello"
    correct = 0
    total = 5
    latencies = []

    for _ in range(total):
        start = time.time()
        try:
            result = subprocess.run(
                ["python", "-c", "print('hello')"],
                capture_output=True, text=True, timeout=5
            )
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)
            if result.stdout.strip().lower() == expected:
                correct += 1
        except Exception:
            latencies.append(5000.0)

    accuracy = correct / total
    latency_ms = sum(latencies) / len(latencies)

    print(json.dumps({"accuracy": accuracy, "latency_ms": round(latency_ms, 2)}))

if __name__ == "__main__":
    evaluate()
