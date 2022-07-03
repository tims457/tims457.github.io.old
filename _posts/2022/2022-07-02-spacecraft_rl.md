---
title: Reinforcement learning for spacecraft
date: 2022-07-02 08:10:00
tags: [space, JAX, reinforcement learning, Flax, CR3BP]
description: Simple reinforcement learning problem for spacecraft in the circular restricted three-body problem (CR3BP)
toc: true
---


<!-- {:class="table-of-content"}
* TOC
{:toc} -->

![training history](/assets/images/2022/rl_for_spacecraft/Animation.gif)

I've been interesting in applying reinforcement learning to spacecraft control problems. Many of the RL problems found online are fairly simple, discrete action space cases, but applying RL to real problems with a continuous action space is more interesting and challenging. 

In the video above, you can see the training progression as an agent learns to transfer from an L1 halo orbit to an L2 halo orbit in the Earth-Moon [circular restricted three-body problem](https://en.wikipedia.org/wiki/Three-body_problem#Restricted_three-body_problem) (CR3BP). The agent computes thrust commands (red arrows) based on its position and velocity vectors. The agent is penalized for getting too close to the moon and rewarded as it comes closer to the target orbit. Starting with initially random actions the agent gradually improves until it starts getting too close to the Moon and has to learn to work with the Moon's gravity to reach the target orbit which is on the other side of the moon from the initial orbit. Thrust was set fairly high to allow the spacecraft to complete the transfer in a shorter amount of simulation time while I worked out the setup and hyperparameters.

# References
The implementation is based on [the work](https://arc.aiaa.org/doi/10.2514/6.2020-1914) by a CU Boulder student, and a PDF is available [here](https://www.colorado.edu/faculty/bosanac/sites/default/files/attached-files/2020_sulbos_aiaa.pdf).  
 

