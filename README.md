# Motivations and Goals

This repo analyzes and plots economics data surrounding the Fed's decision to adjust the Fed funds rate across the last 30 years.

Besides the obvious goal of explaining patterns in the data, this talk is motivated by a few meta-questions:

1. What are good practices for interacting with temporal decision making data? This is motivated by my own work in temporal decision making in the setting of cancer treatment.
2. Can we develop good visual language rules for displaying temporal decision making data in static 2D, interactive 2D, and even 3D formats?
3. How do we quickly understand data, ask creative questions from data, and extract interesting insights using the agentic AI toolkit available in September 2026?

As a part of this exercise:

- I will not be writing a single line of code, nor editing plots. I will strive to not look at the code at all and instead give feedback directly based off of summarized results.
- I will co-write prose with AI in some form or another.

## Data

Our primary data source is the [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/), maintained by the Federal Reserve Bank of St. Louis. It contains an extensive collection of economic data from public and private sources, a majority which are time series, making it well-suited for showcasing the trajectory of the state of the economy over time. [`fredapi`](https://github.com/mortada/fredapi) is a popular python wrapper for accessing the data.

Policy lies at the intersection of science, politics, and society. Besides the quantitive (and semi-quantitive) data from the primary source, we will also use qualitative text from major world events, which may have impacted economic policy decisions. As a simple representation of socio-political events, I've picked Wikipedia's [list of major economic crisis](https://en.wikipedia.org/wiki/List_of_economic_crises#20th_century) and Nikolas Müller-Plantenberg's [list of macroeconomic events](https://www.mullerpl.net/econ/t/int_macro_db/Economic_events.html). It is worth noting upfront that _predictive models_ of Fed rate decisions may do just as well from just the quantitative data alone, but that misses a key point - for the 12 Fed chairs casting their votes in a room, these contemporary events probably has a much larger influence than any single quantitative data track.

## Idea Tree

We will synthesize a visual artifact showing the dynamics of ideas starting, branching, developing, and concluding during the coarse of work. This will be build from agent transcripts and commit history.
