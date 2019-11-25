# API for reinforcement learning
This repositor was created for the Stanford course CME 241, "Reinforcement Learning for Stochastic Control Problems in Finance". 

# Algorithms
Here we cover some of the basic RL algorithms such as temporal difference with difference parameters and reinforce.
Moreover, we also cover the option implementation that is the basis of the small project to replicate the Black-Scholes price with a Leaste Square Policy Iteration (LSPI).

# Examples
Here we cover twoa few examples that can be solved with RL and dynamic progragmming (the latter requires knowledge about the distribution though). 
These examples are:
- Mertons portfolio problem whose goal is to maximize expected utility of an initial endowment throught a particular time horizon
- A car rental which needs to move cars from a one parking lot to another at the end of every night.
- A simple gambler playing a p vs (1-p) coin tossing game who wants to maximize the expected profits
- A grid world which is a simple maze an agent has to uncover

# Processes
Here we use Markov chains for a setup for dynamic programmming. We have A simple markov chain, then a markov reward process and finally a markov decision proceess. Please see David Silver's slides on http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html for a clear definition of all of these.

