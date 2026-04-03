"""Shared pytest fixtures for gold trading tests."""

import pytest


@pytest.fixture
def webhook_secret() -> str:
    return "test_secret_123"


@pytest.fixture
def account_size() -> float:
    return 50_000.0
