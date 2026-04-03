-- 002_fix_macro_precision.sql: Widen cpi_yoy to handle raw CPI index values
-- CPIAUCSL from FRED returns the index level (~327), not YoY percentage

ALTER TABLE macro_data ALTER COLUMN cpi_yoy TYPE NUMERIC(10,4);
