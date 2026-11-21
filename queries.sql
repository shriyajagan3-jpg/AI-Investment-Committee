-- queries.sql - starter queries for committee.db
-- Run all:   sqlite3 committee.db < queries.sql
-- Or open the database and type queries yourself:   sqlite3 committee.db
--   (inside sqlite3: .tables  .schema runs  .mode column  .headers on  .quit)

.mode column
.headers on

-- 1. Every decision, best base-case IRR first  (SELECT, ORDER BY)
SELECT ticker, company, base_irr_pct, final_irr_pct, decision, conviction
FROM runs
ORDER BY base_irr_pct DESC;

-- 2. How many INVEST vs PASS?  (GROUP BY, COUNT)
SELECT decision, COUNT(*) AS companies, ROUND(AVG(base_irr_pct), 1) AS avg_base_irr
FROM runs
GROUP BY decision;

-- 3. Fact-checker accuracy across all memos  (SUM, arithmetic)
SELECT SUM(numbers_verified) AS verified,
       SUM(numbers_checked)  AS checked,
       ROUND(100.0 * SUM(numbers_verified) / SUM(numbers_checked), 1) AS pct_verified
FROM runs;

-- 4. Memos where the fact-checker caught invented numbers  (WHERE)
SELECT ticker, numbers_checked - numbers_verified AS unverified_numbers
FROM runs
WHERE numbers_verified < numbers_checked
ORDER BY unverified_numbers DESC;

-- 5. Which agent runs the most pessimistic scenarios?  (JOIN-free GROUP BY on the second table)
SELECT agent, COUNT(*) AS scenarios_run, ROUND(AVG(irr_pct), 1) AS avg_irr
FROM scenarios
GROUP BY agent
ORDER BY avg_irr;

-- 6. Each company's worst stress test and who ran it  (JOIN + subquery)
SELECT r.ticker, s.agent, s.name AS scenario, s.irr_pct
FROM scenarios s
JOIN runs r ON r.id = s.run_id
WHERE s.irr_pct = (SELECT MIN(irr_pct) FROM scenarios WHERE run_id = s.run_id)
ORDER BY s.irr_pct;

-- 7. How often did agents try to assume a cheaper price or rate?  (JOIN, SUM of a flag)
SELECT r.ticker, SUM(s.market_floor_hit) AS attempts_blocked
FROM runs r
JOIN scenarios s ON s.run_id = r.id
GROUP BY r.ticker
HAVING attempts_blocked > 0;

-- 8. Do higher-margin businesses get better returns?  (CASE buckets)
SELECT CASE WHEN ebitda_margin_pct >= 25 THEN '25%+'
            WHEN ebitda_margin_pct >= 15 THEN '15-25%'
            ELSE 'under 15%' END AS margin_bucket,
       COUNT(*) AS companies,
       ROUND(AVG(base_irr_pct), 1) AS avg_base_irr
FROM runs
GROUP BY margin_bucket
ORDER BY avg_base_irr DESC;
