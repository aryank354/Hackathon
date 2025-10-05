-- =====================================================
-- PRODUCTION SQL QUERIES FOR FLIGHT DIFFICULTY SCORING
-- Team: Innov8torX
-- Purpose: Daily automated scoring pipeline
-- =====================================================

-- =====================================================
-- QUERY 1: DAILY FEATURE EXTRACTION
-- Runs every morning to prepare features for that day's flights
-- =====================================================

WITH flight_base AS (
    SELECT 
        f.company_id,
        f.flight_number,
        f.scheduled_departure_date_local,
        f.scheduled_departure_station_code,
        f.scheduled_arrival_station_code,
        f.scheduled_departure_datetime_local,
        f.scheduled_arrival_datetime_local,
        f.total_seats,
        f.fleet_type,
        f.scheduled_ground_time_minutes,
        f.minimum_turn_minutes,
        EXTRACT(HOUR FROM f.scheduled_departure_datetime_local) AS departure_hour,
        -- Ground time pressure
        CASE 
            WHEN f.scheduled_ground_time_minutes > 0 
            THEN f.minimum_turn_minutes / NULLIF(f.scheduled_ground_time_minutes, 0)
            ELSE 2.0 
        END AS ground_time_pressure,
        -- Insufficient ground time flag
        CASE 
            WHEN f.scheduled_ground_time_minutes < f.minimum_turn_minutes THEN 1 
            ELSE 0 
        END AS insufficient_ground_time
    FROM flight_level_data f
    WHERE f.scheduled_departure_date_local = CURRENT_DATE  -- Today's flights only
),

passenger_summary AS (
    SELECT 
        pnr.company_id,
        pnr.flight_number,
        pnr.scheduled_departure_date_local,
        SUM(pnr.total_pax) AS total_passengers,
        SUM(pnr.lap_child_count) AS total_lap_children,
        SUM(CASE WHEN pnr.is_child = 'Y' THEN 1 ELSE 0 END) AS total_children,
        SUM(pnr.basic_economy_pax) AS total_basic_economy,
        SUM(CASE WHEN pnr.is_stroller_user = 'Y' THEN 1 ELSE 0 END) AS total_stroller_users
    FROM pnr_flight_level_data pnr
    WHERE pnr.scheduled_departure_date_local = CURRENT_DATE
    GROUP BY pnr.company_id, pnr.flight_number, pnr.scheduled_departure_date_local
),

baggage_summary AS (
    SELECT 
        b.company_id,
        b.flight_number,
        b.scheduled_departure_date_local,
        COUNT(b.bag_tag_unique_number) AS total_bags,
        SUM(CASE WHEN b.bag_type = 'Transfer' THEN 1 ELSE 0 END) AS transfer_bags
    FROM bag_level_data b
    WHERE b.scheduled_departure_date_local = CURRENT_DATE
    GROUP BY b.company_id, b.flight_number, b.scheduled_departure_date_local
),

ssr_summary AS (
    SELECT 
        pr.flight_number,
        COUNT(pr.special_service_request) AS total_ssr
    FROM pnr_remark_level_data pr
    WHERE pr.flight_number IN (
        SELECT DISTINCT flight_number 
        FROM flight_level_data 
        WHERE scheduled_departure_date_local = CURRENT_DATE
    )
    GROUP BY pr.flight_number
),

historical_performance AS (
    SELECT 
        f.company_id,
        f.flight_number,
        f.scheduled_departure_station_code,
        f.scheduled_arrival_station_code,
        AVG(EXTRACT(EPOCH FROM (f.actual_departure_datetime_local - f.scheduled_departure_datetime_local)) / 60) AS avg_historical_delay
    FROM flight_level_data f
    WHERE f.scheduled_departure_date_local < CURRENT_DATE
      AND f.scheduled_departure_date_local >= CURRENT_DATE - INTERVAL '30 days'
      AND f.actual_departure_datetime_local IS NOT NULL
    GROUP BY f.company_id, f.flight_number, f.scheduled_departure_station_code, f.scheduled_arrival_station_code
)

SELECT 
    fb.company_id,
    fb.flight_number,
    fb.scheduled_departure_date_local,
    fb.scheduled_departure_station_code,
    fb.scheduled_arrival_station_code,
    fb.scheduled_departure_datetime_local,
    
    -- Ground Operations Features
    fb.ground_time_pressure,
    fb.insufficient_ground_time,
    
    -- Passenger Features
    COALESCE(ps.total_passengers, 0) AS total_passengers,
    COALESCE(ps.total_passengers, 0) / NULLIF(fb.total_seats, 0) AS load_factor,
    COALESCE(ps.total_lap_children, 0) + 
    COALESCE(ps.total_children, 0) * 0.5 + 
    COALESCE(ps.total_stroller_users, 0) * 0.7 AS family_complexity,
    
    -- Baggage Features
    COALESCE(bs.total_bags, 0) / NULLIF(COALESCE(ps.total_passengers, 1), 0) AS bags_per_pax,
    COALESCE(bs.transfer_bags, 0) / NULLIF(COALESCE(bs.total_bags, 1), 0) AS transfer_bag_pct,
    
    -- Special Services
    COALESCE(ssr.total_ssr, 0) / NULLIF(COALESCE(ps.total_passengers, 1), 0) AS ssr_intensity,
    
    -- Historical Performance
    COALESCE(hp.avg_historical_delay, 0) AS historical_delay,
    
    -- Time-based Features
    CASE WHEN fb.departure_hour BETWEEN 6 AND 9 OR fb.departure_hour BETWEEN 16 AND 20 
         THEN 1 ELSE 0 END AS is_peak_hour,
    
    -- Aircraft Size
    CASE 
        WHEN fb.total_seats <= 50 THEN 1
        WHEN fb.total_seats <= 150 THEN 2
        WHEN fb.total_seats <= 300 THEN 3
        ELSE 4
    END AS aircraft_size,
    
    -- International Flag
    CASE WHEN a.iso_country_code != 'US' THEN 1 ELSE 0 END AS is_international,
    
    -- High Stress Flag
    CASE WHEN 
        (COALESCE(ps.total_passengers, 0) / NULLIF(fb.total_seats, 1) > 0.85) AND
        fb.ground_time_pressure > 1 AND
        (fb.departure_hour BETWEEN 6 AND 9 OR fb.departure_hour BETWEEN 16 AND 20)
        THEN 1 ELSE 0 
    END AS high_stress_flight

FROM flight_base fb
LEFT JOIN passenger_summary ps 
    ON fb.company_id = ps.company_id 
    AND fb.flight_number = ps.flight_number 
    AND fb.scheduled_departure_date_local = ps.scheduled_departure_date_local
LEFT JOIN baggage_summary bs 
    ON fb.company_id = bs.company_id 
    AND fb.flight_number = bs.flight_number 
    AND fb.scheduled_departure_date_local = bs.scheduled_departure_date_local
LEFT JOIN ssr_summary ssr 
    ON fb.flight_number = ssr.flight_number
LEFT JOIN historical_performance hp 
    ON fb.company_id = hp.company_id 
    AND fb.flight_number = hp.flight_number
    AND fb.scheduled_departure_station_code = hp.scheduled_departure_station_code
    AND fb.scheduled_arrival_station_code = hp.scheduled_arrival_station_code
LEFT JOIN airports_data a 
    ON fb.scheduled_arrival_station_code = a.airport_iata_code
ORDER BY fb.scheduled_departure_datetime_local;


-- =====================================================
-- QUERY 2: DAILY PERCENTILE RANKING
-- Calculates percentile ranks for each feature within the day
-- =====================================================

WITH daily_features AS (
    -- Insert results from Query 1 above
    -- This would come from a temp table or materialized view
    SELECT * FROM daily_flight_features  -- Output from Query 1
),

feature_ranks AS (
    SELECT 
        *,
        PERCENT_RANK() OVER (ORDER BY ground_time_pressure) AS ground_time_pressure_rank,
        PERCENT_RANK() OVER (ORDER BY insufficient_ground_time) AS insufficient_ground_time_rank,
        PERCENT_RANK() OVER (ORDER BY load_factor) AS load_factor_rank,
        PERCENT_RANK() OVER (ORDER BY family_complexity) AS family_complexity_rank,
        PERCENT_RANK() OVER (ORDER BY bags_per_pax) AS bags_per_pax_rank,
        PERCENT_RANK() OVER (ORDER BY transfer_bag_pct) AS transfer_bag_pct_rank,
        PERCENT_RANK() OVER (ORDER BY ssr_intensity) AS ssr_intensity_rank,
        PERCENT_RANK() OVER (ORDER BY historical_delay) AS historical_delay_rank,
        PERCENT_RANK() OVER (ORDER BY is_peak_hour) AS is_peak_hour_rank,
        PERCENT_RANK() OVER (ORDER BY aircraft_size) AS aircraft_size_rank,
        PERCENT_RANK() OVER (ORDER BY is_international) AS is_international_rank,
        PERCENT_RANK() OVER (ORDER BY high_stress_flight) AS high_stress_flight_rank
    FROM daily_features
)

SELECT 
    company_id,
    flight_number,
    scheduled_departure_date_local,
    scheduled_departure_station_code,
    scheduled_arrival_station_code,
    scheduled_departure_datetime_local,
    
    -- Original features
    ground_time_pressure,
    load_factor,
    family_complexity,
    bags_per_pax,
    transfer_bag_pct,
    ssr_intensity,
    historical_delay,
    is_peak_hour,
    aircraft_size,
    is_international,
    high_stress_flight,
    
    -- Calculate composite difficulty score
    (ground_time_pressure_rank + insufficient_ground_time_rank + 
     load_factor_rank + family_complexity_rank + bags_per_pax_rank + 
     transfer_bag_pct_rank + ssr_intensity_rank + historical_delay_rank + 
     is_peak_hour_rank + aircraft_size_rank + is_international_rank + 
     high_stress_flight_rank) / 12.0 * 100 AS difficulty_score,
    
    -- Daily rank
    RANK() OVER (ORDER BY 
        (ground_time_pressure_rank + insufficient_ground_time_rank + 
         load_factor_rank + family_complexity_rank + bags_per_pax_rank + 
         transfer_bag_pct_rank + ssr_intensity_rank + historical_delay_rank + 
         is_peak_hour_rank + aircraft_size_rank + is_international_rank + 
         high_stress_flight_rank) DESC
    ) AS daily_rank,
    
    -- Percentile within day
    PERCENT_RANK() OVER (ORDER BY 
        (ground_time_pressure_rank + insufficient_ground_time_rank + 
         load_factor_rank + family_complexity_rank + bags_per_pax_rank + 
         transfer_bag_pct_rank + ssr_intensity_rank + historical_delay_rank + 
         is_peak_hour_rank + aircraft_size_rank + is_international_rank + 
         high_stress_flight_rank) DESC
    ) AS daily_percentile,
    
    -- Difficulty classification
    CASE 
        WHEN PERCENT_RANK() OVER (ORDER BY 
            (ground_time_pressure_rank + insufficient_ground_time_rank + 
             load_factor_rank + family_complexity_rank + bags_per_pax_rank + 
             transfer_bag_pct_rank + ssr_intensity_rank + historical_delay_rank + 
             is_peak_hour_rank + aircraft_size_rank + is_international_rank + 
             high_stress_flight_rank) DESC
        ) <= 0.20 THEN 'Difficult'
        WHEN PERCENT_RANK() OVER (ORDER BY 
            (ground_time_pressure_rank + insufficient_ground_time_rank + 
             load_factor_rank + family_complexity_rank + bags_per_pax_rank + 
             transfer_bag_pct_rank + ssr_intensity_rank + historical_delay_rank + 
             is_peak_hour_rank + aircraft_size_rank + is_international_rank + 
             high_stress_flight_rank) DESC
        ) <= 0.70 THEN 'Medium'
        ELSE 'Easy'
    END AS difficulty_class

FROM feature_ranks
ORDER BY daily_rank;


-- =====================================================
-- QUERY 3: TOP 10 DIFFICULT FLIGHTS ALERT
-- Daily morning report for operations team
-- =====================================================

WITH scored_flights AS (
    -- Results from Query 2
    SELECT * FROM daily_scored_flights
)

SELECT 
    daily_rank,
    flight_number,
    scheduled_departure_station_code || ' → ' || scheduled_arrival_station_code AS route,
    TO_CHAR(scheduled_departure_datetime_local, 'HH24:MI') AS departure_time,
    difficulty_score,
    difficulty_class,
    -- Alert reasons
    CASE WHEN insufficient_ground_time = 1 THEN '⚠️ Tight Turn' ELSE '' END ||
    CASE WHEN load_factor > 0.85 THEN ' 🎫 High Load' ELSE '' END ||
    CASE WHEN is_peak_hour = 1 THEN ' ⏰ Peak Hour' ELSE '' END ||
    CASE WHEN transfer_bag_pct > 0.30 THEN ' 🧳 High Transfer %' ELSE '' END ||
    CASE WHEN ssr_intensity > 0.05 THEN ' ♿ High SSR' ELSE '' END ||
    CASE WHEN is_international = 1 THEN ' 🌍 International' ELSE '' END AS alert_flags,
    
    -- Recommended actions
    CASE 
        WHEN insufficient_ground_time = 1 AND transfer_bag_pct > 0.30 
            THEN 'Priority: Pre-position baggage team'
        WHEN load_factor > 0.85 AND is_peak_hour = 1 
            THEN 'Priority: Open extra gates early'
        WHEN ssr_intensity > 0.05 
            THEN 'Action: Assign dedicated assistance staff'
        WHEN is_international = 1 AND load_factor > 0.85 
            THEN 'Action: Additional customs support'
        ELSE 'Monitor closely'
    END AS recommended_action

FROM scored_flights
WHERE difficulty_class = 'Difficult'
ORDER BY daily_rank
LIMIT 10;


-- =====================================================
-- QUERY 4: OPERATIONAL DASHBOARD METRICS
-- Real-time monitoring during the day
-- =====================================================

WITH todays_flights AS (
    SELECT * FROM daily_scored_flights
)

SELECT 
    'Total Flights Today' AS metric,
    COUNT(*)::TEXT AS value,
    '' AS comparison
FROM todays_flights

UNION ALL

SELECT 
    'Difficult Flights',
    COUNT(*)::TEXT,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM todays_flights), 1)::TEXT || '%'
FROM todays_flights
WHERE difficulty_class = 'Difficult'

UNION ALL

SELECT 
    'Avg Difficulty Score',
    ROUND(AVG(difficulty_score), 1)::TEXT,
    ''
FROM todays_flights

UNION ALL

SELECT 
    'Flights with Tight Turns',
    SUM(insufficient_ground_time)::TEXT,
    ROUND(SUM(insufficient_ground_time) * 100.0 / COUNT(*), 1)::TEXT || '%'
FROM todays_flights

UNION ALL

SELECT 
    'Peak Hour Operations',
    SUM(is_peak_hour)::TEXT,
    ROUND(SUM(is_peak_hour) * 100.0 / COUNT(*), 1)::TEXT || '%'
FROM todays_flights;


-- =====================================================
-- QUERY 5: WEEKLY TREND ANALYSIS
-- Performance tracking over time
-- =====================================================

SELECT 
    DATE_TRUNC('week', scheduled_departure_date_local) AS week_start,
    COUNT(*) AS total_flights,
    SUM(CASE WHEN difficulty_class = 'Difficult' THEN 1 ELSE 0 END) AS difficult_flights,
    ROUND(AVG(difficulty_score), 1) AS avg_difficulty_score,
    ROUND(AVG(CASE WHEN difficulty_class = 'Difficult' 
                   THEN EXTRACT(EPOCH FROM (actual_departure_datetime_local - scheduled_departure_datetime_local)) / 60 
                   END), 1) AS avg_difficult_delay_minutes
FROM daily_scored_flights
WHERE scheduled_departure_date_local >= CURRENT_DATE - INTERVAL '8 weeks'
  AND actual_departure_datetime_local IS NOT NULL
GROUP BY DATE_TRUNC('week', scheduled_departure_date_local)
ORDER BY week_start DESC;


-- =====================================================
-- QUERY 6: ROUTE-SPECIFIC DIFFICULTY PATTERNS
-- Identify consistently difficult routes
-- =====================================================

SELECT 
    scheduled_arrival_station_code AS destination,
    COUNT(*) AS total_flights,
    ROUND(AVG(difficulty_score), 1) AS avg_difficulty,
    SUM(CASE WHEN difficulty_class = 'Difficult' THEN 1 ELSE 0 END) AS difficult_count,
    ROUND(SUM(CASE WHEN difficulty_class = 'Difficult' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS difficult_pct,
    ROUND(AVG(load_factor), 2) AS avg_load_factor,
    ROUND(AVG(transfer_bag_pct), 2) AS avg_transfer_pct
FROM daily_scored_flights
WHERE scheduled_departure_date_local >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY scheduled_arrival_station_code
HAVING COUNT(*) >= 10  -- Only routes with sufficient data
ORDER BY avg_difficulty DESC
LIMIT 20;


-- =====================================================
-- QUERY 7: PREDICTIVE ALERT SYSTEM
-- Flag flights likely to have issues 24 hours in advance
-- =====================================================

WITH tomorrows_flights AS (
    SELECT * FROM daily_scored_flights
    WHERE scheduled_departure_date_local = CURRENT_DATE + INTERVAL '1 day'
)

SELECT 
    flight_number,
    scheduled_departure_datetime_local,
    scheduled_arrival_station_code AS destination,
    difficulty_score,
    difficulty_class,
    
    -- Risk factors
    JSON_BUILD_OBJECT(
        'tight_ground_time', insufficient_ground_time = 1,
        'high_load', load_factor > 0.90,
        'peak_hour', is_peak_hour = 1,
        'high_transfer_bags', transfer_bag_pct > 0.30,
        'high_ssr', ssr_intensity > 0.05,
        'international', is_international = 1,
        'historical_delays', historical_delay > 20
    ) AS risk_factors,
    
    -- Confidence score (based on data completeness)
    CASE 
        WHEN total_passengers > 0 AND total_bags > 0 AND total_ssr >= 0 
        THEN 'High'
        WHEN total_passengers > 0 
        THEN 'Medium'
        ELSE 'Low'
    END AS prediction_confidence

FROM tomorrows_flights
WHERE difficulty_class = 'Difficult'
ORDER BY difficulty_score DESC;