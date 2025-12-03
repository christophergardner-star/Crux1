import json
import time

class MetricsLogger:
    """
    Logs training metrics to a JSONL file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        # Clear file or append? Usually append or overwrite.
        # We'll overwrite for new run.
        with open(self.file_path, 'w') as f:
            pass

    def log(self, metrics):
        metrics['timestamp'] = time.time()
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
