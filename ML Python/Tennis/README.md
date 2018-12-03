<img src="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif">


# DRL - MADDPG Algorithm - Tennis Collaboration Continuous Control
Udacity Deep Reinforcement Learning Nanodegree Program - Tennis Collaboration Continuous Control


### Observations:
- To run the project just execute the <b>main.py</b> file.
- There is also an .ipynb file for jupyter notebook execution.
- If you are not using a windows environment, you will need to download the corresponding <b>"Tennis"</b> version for you OS system. Mail me if you need more details about the environment <b>.exe</b> file.
- The <b>checkpoint.pth</b> has the expected average score already hit.


### Requeriments:
- tensorflow: 1.7.1
- Pillow: 4.2.1
- matplotlib
- numpy: 1.11.0
- pytest: 3.2.2
- docopt
- pyyaml
- protobuf: 3.5.2
- grpcio: 1.11.0
- torch: 0.4.1
- pandas
- scipy
- ipykernel
- jupyter: 5.6.0


## The problem:
- The task solved here refers to a collaboration continuous control problem where two agents must be able to play "tennis"
in collaboration, that is, the longer the rally goes the higher will be the reward that both agents will earn.
- It's a continuous problem because the action has a continuous value and the agents must be able to provide this value instead of just chose the one with the biggest value (like in discrete tasks where it should just say which action it wants to execute).
- In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
- The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).


## The solution:
- For this problem, I used an implementation of mine from the Multi-Agent Deep Deterministic Policy Gradients algorithm (the DDPG code provided by Udacity was used as a reference).
- The challenge here was to find the best way of sharing the experiences between the agents. In the first version of the algorithm, I had two actors, two critics, and a shared memory buffer. I don't know why exactly this idea doesn't work as I expected and then, after a few more research, I migrated my implementation aiming to separate the critic from the actors, that is, one critic was shared by both actors and each actor now had its memory buffer. This idea worked but then I realized that the task was the same (just from a different perspective) and that I could not only share the critic but also the experiences collected by all the actors. I made this change and after hyperparameters tuning, the solution reached the actual results.
- I also tested some variations of noise like increasing the noise range and a noise reduction over time approach, but I didn't get better results, so I removed it by now.
- I've noticed that the convergence is still a little bit unstable and for the future, I plan to make an improvement upon the neural network structure and check if I can have a faster convergence for this task.


### The hyperparameters:
- The file with the hyperparameters configuration is the <b>main.py</b>. 
- If you want you can change the model configuration to into the <b>model.py</b> file.
- The noise configuration is in the <b>noise.py</b> file and the values are fixed in the method signature.
- The actual configuration of the hyperparameters is: 
  - Learning Rate:
    - Actors: 1e-4
    - Critic: 3e-4
  - Batch Size: 128
  - Replay Buffer: 1e6
  - Gamma: 0.99
  - Tau: 2e-1
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)

- For the neural models:    
  - Actor    
    - Hidden: (input, 512)  - ReLU
    - Hidden: (512, 256)    - ReLU
    - Output: (256, 2)      - TanH

  - Critic
    - Hidden: (input, 512)              - ReLU
    - Hidden: (512 + action_size, 256)  - ReLU
    - Output: (256, 1)                  - Linear
