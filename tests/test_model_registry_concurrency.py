# tests/test_model_registry_concurrency.py
import threading
import time

import pytest

from src.apps.app import ModelRegistry


class DummyPredictor:
    """Simple fake predictor used only for concurrency testing."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, text: str):
        # We don't really use this in the test, but it mimics the real interface.
        return 0, 1.0


def test_model_registry_concurrent_predict_and_switch():
    """
    This test verifies that ModelRegistry is safe under concurrent access.

    We simulate:
      - Several threads repeatedly calling get_model_context() (like /predict).
      - One thread repeatedly calling switch_model() (like /models/switch).

    The expectation:
      - No exceptions are raised.
      - Whenever a predictor is returned by get_model_context(), it is not None.
    """

    registry = ModelRegistry()

    # Register two dummy models with different predictors.
    predictor_a = DummyPredictor("A")
    predictor_b = DummyPredictor("B")

    registry.register_model("model_a", object(), predictor_a, "LSTM")
    registry.register_model("model_b", object(), predictor_b, "BERT")

    stop_event = threading.Event()
    errors = []

    def predicting_worker():
        """Continuously fetch predictors and access them."""
        while not stop_event.is_set():
            try:
                with registry.get_model_context() as predictor:
                    # After at least one model is registered, predictor should never be None
                    assert predictor is not None
                    # Access an attribute to ensure the object is valid
                    _ = predictor.name
            except Exception as e:
                # Record the first error and stop all threads
                errors.append(e)
                stop_event.set()

    def switching_worker():
        """Continuously switch between the two models."""
        paths = ["model_a", "model_b"]
        idx = 0
        while not stop_event.is_set():
            try:
                registry.switch_model(paths[idx % 2])
                idx += 1
            except Exception as e:
                errors.append(e)
                stop_event.set()

    # Start multiple predictor threads
    predictor_threads = [
        threading.Thread(target=predicting_worker)
        for _ in range(4)
    ]

    for t in predictor_threads:
        t.start()

    # Start one switcher thread
    switcher_thread = threading.Thread(target=switching_worker)
    switcher_thread.start()

    # Let them run concurrently for a short period
    time.sleep(5)
    stop_event.set()

    # Wait for all threads to finish
    for t in predictor_threads:
        t.join()
    switcher_thread.join()

    # If any exception occurred, fail the test and show it
    assert not errors, f"Concurrency errors occurred: {errors}"
