"""Train the GaussianHMM regime detection model on historical GC 5m data.

Trains a 4-state HMM on features: [log_return, atr_14, adx_14].
Saves the model + SHA-256 hash for secure loading by the Regime Analyst.

Usage:
    uv run python scripts/train_hmm.py

Re-run periodically (monthly) as more data accumulates to keep the model fresh.
"""

import asyncio
import hashlib
import pickle
from pathlib import Path

import asyncpg
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from gold_trading.db.client import get_database_url

MODEL_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = MODEL_DIR / "hmm_regime_model.pkl"
HASH_PATH = MODEL_DIR / "hmm_regime_model.sha256"

N_STATES = 4  # trending_up, trending_down, ranging, volatile
N_ITER = 200
RANDOM_SEED = 42


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate HMM input features from OHLCV data."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Log returns
    df["log_return"] = np.log(close / close.shift(1))

    # ATR(14)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ADX(14)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index
    )
    atr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_smooth)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df["adx_14"] = dx.rolling(14).mean()

    # Rolling volatility (20-bar)
    df["volatility_20"] = df["log_return"].rolling(20).std()

    return df.dropna()


def label_states(model: GaussianHMM, features: np.ndarray) -> dict[int, str]:
    """Map HMM states to human-readable regime labels.

    Uses the state means to identify:
    - Highest mean return + low vol → trending_up
    - Lowest mean return + low vol → trending_down
    - Low ADX + low vol → ranging
    - Highest vol → volatile
    """
    means = model.means_  # Shape: (n_states, n_features)
    # Features: [log_return, atr_14, adx_14, volatility_20]

    labels = {}
    assigned = set()

    # Volatile: highest volatility (feature index 3)
    vol_idx = int(np.argmax(means[:, 3]))
    labels[vol_idx] = "volatile"
    assigned.add(vol_idx)

    # Trending up: highest mean return among remaining
    remaining = [i for i in range(N_STATES) if i not in assigned]
    up_idx = max(remaining, key=lambda i: means[i, 0])
    labels[up_idx] = "trending_up"
    assigned.add(up_idx)

    # Trending down: lowest mean return among remaining
    remaining = [i for i in range(N_STATES) if i not in assigned]
    down_idx = min(remaining, key=lambda i: means[i, 0])
    labels[down_idx] = "trending_down"
    assigned.add(down_idx)

    # Ranging: whatever's left
    remaining = [i for i in range(N_STATES) if i not in assigned]
    labels[remaining[0]] = "ranging"

    return labels


async def load_training_data() -> pd.DataFrame:
    """Load all 5m OHLCV data for training."""
    conn = await asyncpg.connect(get_database_url())
    try:
        rows = await conn.fetch(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_5m WHERE instrument = 'GC'
            ORDER BY timestamp ASC
            """
        )
        if not rows:
            raise RuntimeError("No OHLCV data in database. Run ingest_ohlcv.py first.")

        df = pd.DataFrame([dict(r) for r in rows])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(int)

        logger.info(f"Loaded {len(df)} bars for training")
        return df

    finally:
        await conn.close()


def train_model(df: pd.DataFrame) -> tuple[GaussianHMM, dict[int, str], dict]:
    """Train the HMM and return model, state labels, and diagnostics."""
    features_df = calculate_features(df)
    feature_cols = ["log_return", "atr_14", "adx_14", "volatility_20"]
    x_data = features_df[feature_cols].values

    logger.info(
        f"Training HMM on {len(x_data)} observations, {N_STATES} states, {len(feature_cols)} features"
    )

    # Normalize features for stable training
    means = x_data.mean(axis=0)
    stds = x_data.std(axis=0)
    stds[stds == 0] = 1  # Prevent division by zero
    x_norm = (x_data - means) / stds

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    model.fit(x_norm)

    # Check convergence
    converged = model.monitor_.converged
    score = model.score(x_norm)
    logger.info(f"HMM converged: {converged}, log-likelihood: {score:.2f}")

    # Predict states
    states = model.predict(x_norm)
    state_probs = model.predict_proba(x_norm)

    # Label states
    labels = label_states(model, x_norm)
    logger.info(f"State labels: {labels}")

    # Diagnostics
    state_counts = {labels.get(i, f"state_{i}"): int(np.sum(states == i)) for i in range(N_STATES)}
    avg_confidence = {
        labels.get(i, f"state_{i}"): float(np.mean(state_probs[states == i, i]))
        for i in range(N_STATES)
        if np.sum(states == i) > 0
    }

    diagnostics = {
        "n_observations": len(x_data),
        "n_states": N_STATES,
        "converged": converged,
        "log_likelihood": float(score),
        "state_counts": state_counts,
        "avg_confidence": avg_confidence,
        "feature_means": {col: float(m) for col, m in zip(feature_cols, means, strict=True)},
        "feature_stds": {col: float(s) for col, s in zip(feature_cols, stds, strict=True)},
    }

    # Store normalization params in the model for inference
    model._norm_means = means
    model._norm_stds = stds
    model._state_labels = labels

    return model, labels, diagnostics


def save_model(model: GaussianHMM) -> None:
    """Save model and its SHA-256 hash."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save hash for security verification
    model_bytes = MODEL_PATH.read_bytes()
    hash_hex = hashlib.sha256(model_bytes).hexdigest()
    HASH_PATH.write_text(hash_hex)

    logger.info(f"Model saved to {MODEL_PATH} (hash: {hash_hex[:16]}...)")


async def main() -> None:
    logger.info("HMM regime model training starting")

    # Load data
    df = await load_training_data()

    if len(df) < 500:
        logger.error(f"Only {len(df)} bars — need at least 500 for meaningful HMM training")
        return

    # Train
    model, _labels, diagnostics = train_model(df)

    # Save
    save_model(model)

    # Report
    logger.info("=== Training Report ===")
    logger.info(f"  Observations: {diagnostics['n_observations']}")
    logger.info(f"  Converged: {diagnostics['converged']}")
    logger.info(f"  Log-likelihood: {diagnostics['log_likelihood']:.2f}")
    for state_name, count in diagnostics["state_counts"].items():
        pct = count / diagnostics["n_observations"] * 100
        conf = diagnostics["avg_confidence"].get(state_name, 0)
        logger.info(f"  {state_name}: {count} bars ({pct:.1f}%), avg confidence: {conf:.3f}")

    logger.info("HMM training complete")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
