-- 003_add_strategy_trade_count.sql: Add trade count and profit factor to strategies

ALTER TABLE strategies ADD COLUMN IF NOT EXISTS vbt_total_trades INTEGER;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS vbt_profit_factor NUMERIC(8,2);
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS vbt_avg_duration_min NUMERIC(8,1);
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS backtest_params JSONB;
