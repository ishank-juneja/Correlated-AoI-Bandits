# Correlated Age-of-Information Bandits

<p align="center">
  <img src="media/system1.png" width="400"/>
  <br>
<b>A regret plot example for an instance where UCUCB outperformed CUCB</b>
</p>

Authors' implementation of [the paper](https://arxiv.org/abs/2011.05032) Correlated Age-of-Information Bandits. 

Video Presentation: [presentation](https://youtu.be/yDz-bNDMiWM)

Presentation Slides: [slides](http://home.iitb.ac.in/~ishankjuneja/files/AoI_bandits.pdf)

By extension, the code also implements the papers -
- Multi-Armed Bandits with Correlated Arms - [paper](https://arxiv.org/abs/1911.03959?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)
- Regret of Age-of-Information Bandits - [paper](https://arxiv.org/abs/2001.09317)

### Abstract
We consider a system composed of a sensor node tracking a time varying quantity. In every discretized time slot, the node attempts to send an update to a central monitoring station through one of K communication channels. We consider the setting where channel realizations are correlated across channels. This is motivated by mmWave based 5G systems where line-of-sight which is critical for successful communication is common across all frequency channels while the effect of other factors like humidity is frequency dependent. 

The metric of interest is the Age-of-Information (AoI) which is a measure of the freshness of the data available at the monitoring station. In the setting where channel statistics are unknown but stationary across time and correlated across channels, the algorithmic challenge is to determine which channel to use in each time-slot for communication. We model the problem as a Multi-Armed bandit (MAB) with channels as arms. We characterize the fundamental limits on the performance of any policy. In addition, via analysis and simulations, we characterize the performance of variants of the UCB and Thompson Sampling policies that exploit correlation. 

### Dependencies
    Python 3.8
Libraries

    numpy
    matplotlib
    argprase

### Usage Instructions
1. Define the bandit instance on which you wish to run the experiment using a `.txt` file placed in `instances`.
   
2. Use the below file format for the correlated bandit instance 
   
        <1> Distribution of the latent state X as a sequence of floats summing to 1.0    
        <2> K lines with the nth line corresponding to the arm function Y_n (X)
For an example please see `instances/i-1.txt` which is the same as the example I_1 used in the paper on Correlated Age-of-Information Bandits.

3. Create a folder called results in the same directory as `simulate_policies.py` and run `./generate_results.sh` to generate plots analogous to the ones in our paper.

### Related

- A New Approach to Correlated Multi-Armed Bandits paper [repository](https://github.com/ishank-juneja/UCUCB)
- Expert policy guided PAC exploration [repository](https://github.com/ishank-juneja/expert-guided-PACexploration)
- Reward search and reward shaping in Reinforcement Learning [repository](https://github.com/ishank-juneja/reward-search-shaping)
