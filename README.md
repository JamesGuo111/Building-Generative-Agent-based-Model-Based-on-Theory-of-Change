# Building-Generative-Agent-based-Model-Based-on-Theory-of-Change

## Overview
This project is developing conceptual framework to build Generative Agent-Based Model (GABM) grounded in the Theory of Change (ToC) framework to improve the evaluation policy interventions. The goal is to bridge traditional agent-based modeling with large language models (LLMs) as decision engines, enabling agents to make context-sensitive, explainable, and heterogeneous decisions while remaining embedded in a structured causal framework. To demonstrate the feasibility of the framework, a technical prototype was constructed based on a case study in the education domain, focusing on the Girls’ Education Challenge, an initiative aimed at improving educational access and outcomes for girls.

Unlike conventional ABMs that rely on fixed behavioral rules, this model equips individual agents (e.g., students, households, schools) with LLM-driven decision modules. Agents perceive their local state, institutional constraints, and intervention signals, and then generate decisions along with natural-language explanations. To ensure computational feasibility and reproducibility, all LLM decisions are cached using a structured decision cache keyed by discretized state variables.

The Theory of Change serves as the backbone of the model design. Outcomes, intermediate states, mechanisms, and assumptions are explicitly encoded into the simulation logic, allowing the model to function not only as a predictive tool but also as a mechanism-exploration and policy-evaluation framework. This structure makes it possible to analyze how micro-level decision processes aggregate into system-level outcomes under different interventions and assumptions.

The project is designed as a modular research prototype. It emphasizes transparency, extensibility, and sensitivity analysis, supporting experiments such as alternative transition rules, intervention strengths, initialization distributions, and the presence or absence of social or institutional constraints. More broadly, the project aims to demonstrate how GABM can complement traditional evaluation methods by combining causal reasoning, bounded rationality, and generative decision-making within a unified simulation framework.

## Core Module
# model.py
model.py defines the core Agent-Based Model that orchestrates the simulation environment, agent lifecycle, decision-making processes, and policy transition dynamics. It serves as the central integration layer connecting agents, institutional settings, learning and transition policies, and LLM-based decision engines.

The model combines rule-based dynamics with LLM-driven decisions. Large language models are used as decision engines for attendance, attitude updates, self-esteem evaluation, and year-end schooling transitions, while policy-controlled mechanisms govern learning accumulation, absence effects, and transition probabilities.

A Theory of Change–inspired structure is embedded in the model design. Institutional conditions (e.g., school quality, fairness, safety, peer support) influence individual decisions, which then aggregate into system-level outcomes such as enrollment, dropout, vocational transitions, employment, and graduation.

To ensure reproducibility and scalability, all LLM-generated decisions are stored in a structured decision cache and reused across similar states. The model proceeds in discrete time steps with clear daily, periodic, and annual phases, supporting transparent analysis of how micro-level decisions translate into long-term educational trajectories.

# translator.py
translator.py provides a translation layer between the numerical state space of the agent-based model and the natural-language inputs required by LLM decision engines.

Internally, the model represents agent and institutional attributes using a unified 0–100 scale (or small discrete sets). The translator maps these numeric values into bucketed, human-readable descriptions based on predefined specifications, while simultaneously returning bucket indices and ranges used to construct structured decision cache keys.

This design ensures consistency between LLM prompts and cached decisions, enabling reproducible and efficient reuse of LLM outputs across similar agent states. The module supports translating individual attributes as well as grouped states (girl, household, school), and formats results directly for prompt assembly.

# prompt_builder.py
prompt_builder.py defines how structured agent states are converted into controlled, schema-consistent natural-language prompts for LLM-based decision-making.

Building on the attribute translations provided by translator.py, this module assembles prompts for key decision points such as attendance, schooling transitions, self-esteem evaluation, and household attitudes. Prompts are carefully constructed to restrict the LLM to feasible options, enforce exact output formats, and maintain consistency between generated decisions and downstream parsing.

By centralizing prompt design, this module ensures that LLM behavior remains interpretable, reproducible, and aligned with the model’s causal structure, while allowing prompt logic to be modified independently of simulation dynamics.

# decision_cache.py
decision_cache.py implements a structured decision caching mechanism that enables scalable and reproducible use of LLMs within the agent-based model.

Because agent attributes are discretized into buckets before being translated into natural language, identical bucket configurations produce identical prompts. This module exploits that structure by caching LLM decisions keyed by bucketed agent and institutional states. When the same decision context reappears, cached results can be reused instead of issuing a new LLM call.

To account for LLM stochasticity, the cache stores multiple samples for each decision key and only begins reuse after a minimum number of samples are collected, forming an empirical distribution over decisions. This design significantly reduces computational cost while preserving behavioral variability and consistency across simulation runs.

# sim_controller.py
sim_controller.py provides the policy and parameter control layer that enables Theory of Change (ToC) analysis within the simulation.

Rather than hard-coding intervention effects, this module explicitly defines how learning processes, transition probabilities, and institutional conditions evolve over time. It centralizes assumptions about causal mechanisms, including how school quality affects learning, how attendance influences progression, and how policy parameters change under different intervention scenarios.

By separating parameter initialization, temporal schedules, learning rules, and transition policies from the main simulation logic, the controller allows the modeler to systematically test alternative causal chains. Running the same model under different parameter evolution schemes makes it possible to evaluate whether intended interventions lead to the expected downstream outcomes.

