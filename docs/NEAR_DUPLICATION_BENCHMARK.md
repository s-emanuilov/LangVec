# Benchmark near duplication with PostgreSQL

## Setup

In this script above, we are setting up a PostgreSQL database to benchmark string similarity search using the
Levenshtein distance metric. Here's a brief explanation of what we're doing exactly:

1. We create a table strings to store 10 million random strings of length 32, consisting of only lowercase letters
   from 'a' to 'z'.
2. We define a function `generate_random_string` to generate random strings of a specified length.
3. We insert 10 million random strings into the strings table using the `generate_random_string` function.
4. We create an index on the str column of the strings table using the `pg_trgm` extension, which provides efficient
   trigram-based string similarity search.
5. We enable the `fuzzystrmatch` extension, which provides the `levenshtein` function for calculating the Levenshtein
   distance between strings.
6. We create a function `find_similar_strings` that takes an input string (input_str) and a maximum Levenshtein
   distance (
   `max_distance`) as parameters, and returns a table of up to 100 strings from the strings table that are within the
   specified Levenshtein distance from the input string, sorted by distance.

```sql
-- Create a table to store strings
CREATE TABLE strings
(
    id  SERIAL PRIMARY KEY,
    str TEXT NOT NULL
);

-- Create a function to generate random strings
CREATE OR REPLACE FUNCTION generate_random_string(length INTEGER)
    RETURNS TEXT AS
$$
DECLARE
    chars  TEXT[] := '{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z}';
    result TEXT   := '';
BEGIN
    FOR i IN 1..length
        LOOP
            result := result || chars[1 + random() * (array_length(chars, 1) - 1)];
        END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Insert 10 million random strings into the table
INSERT INTO strings (str)
SELECT generate_random_string(32)
FROM generate_series(1, 10000000);

-- Create an index on the str column for faster similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX strings_trgm_idx ON strings USING GIN (str gin_trgm_ops);

-- Create an extension to enable the Levenshtein distance function
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- Create a function to find similar strings
CREATE OR REPLACE FUNCTION find_similar_strings(input_str TEXT, max_distance INTEGER)
    RETURNS TABLE
            (
                id       INTEGER,
                str      TEXT,
                distance INTEGER
            )
AS
$$
BEGIN
    RETURN QUERY
        SELECT s.id, s.str, levenshtein(input_str, s.str) AS distance
        FROM strings s
        WHERE levenshtein(input_str, s.str) <= max_distance
        ORDER BY distance
        LIMIT 100;
END;
$$ LANGUAGE plpgsql;
```

## Usage

To use this setup for benchmarking string similarity search, you can follow these steps:

1. First, we generate a random input string of length 32 using the `generate_random_string` function.
2. Then, we call the `find_similar_strings` function with the generated input string and a maximum Levenshtein distance
   of 3.
3. This function will return up to 10 strings from the strings table that have a Levenshtein distance of 30 or less from
   the input string, along with their respective distances.

```sql
-- Generate a random input string
SELECT generate_random_string(32) AS input_str;
-- Output: 'abcdefghijklmnopqrstuvwxyz012345'

-- Measure the time to find similar strings
SELECT str, distance
FROM find_similar_strings('abcdefghijklmnopqrstuvwxyz012345', 30)
ORDER BY distance
LIMIT 10;
```

By measuring the execution time of the `find_similar_strings` function with different input strings and maximum distances,
you can benchmark the performance of string similarity search in PostgreSQL. 