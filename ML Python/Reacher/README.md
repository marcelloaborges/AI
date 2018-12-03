<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif">


# DRL - DDPG Algorithm - Reacher Continuous Control
Udacity Deep Reinforcement Learning Nanodegree Program - Reacher Continuous Control


### Observations:
- To run the project just execute the <b>main.py</b> file.
- There is also an .ipynb file for jupyter notebook execution.
- If you are not using a windows environment, you will need to download the corresponding <b>"Reacher"</b> version for you OS system. Mail me if you need more details about the environment <b>.exe</b> file.
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
- The task solved here refers to a continuous control problem where the agent must be able to reach and go along with a moving ball controlling its arms.
- It's a continuous problem because the action has a continuous value and the agent must be able to provide this value instead of just chose the one with the biggest value (like in discrete tasks where it should just say which action it wants to execute).
- The reward of +0.1 is provided for each step that the agent's hand is in the goal location, in this case, the moving ball.
- The environment provides 2 versions, one with just 1 agent and another one with 20 agents working in parallel.
- For both versions the goal is to get an average score of +30 over 100 consecutive episodes (for the second version, the average score of all agents must be +30).


## The solution:
- For this problem I used an implementation of the Deep Deterministic Policy Gradients algorithm.
- This task brought two big challenges for me: hyperparameters tunning and noise range configuration. After I found the right configuration for these two points the solution worked impressively well. I must say that the noise range configuration is the key for this task. As the action is a continuous value, dealing with noise correctly means more generalization and makes the agent convergence faster and more robust. The other hyperparameters increase the convergence speed but almost never prevent the agent from finding the solution whereas the wrong noise range configuration can easily make the agent unstable and, I risk saying, impossible to converge.
- Another thing to highlight here is how great the approach used in actor critic structures in general is. It really takes the good part of both worlds, value based methods and policy gradient methods, and makes them work together in an impressive way. Especially in this task, the way the actor and critic learn together sharing their experiences really brought to my eyes a revolutionary point of view about how to build machine learning algorithms. It's really worth to take a look.
- For the future, although the actual solution seems pretty good to me, I stil want to check this task with the D4PG algorithm and discover when and where each of the algorithms (DDPG vs. D4PG) have the best performance.


### The hyperparameters:
- The file with the hyperparameters configuration is the <b>main.py</b>. 
- If you want you can change the model configuration to into the <b>model.py</b> file.
- The actual configuration of the hyperparameters is: 
  - Learning Rate: 1e-4 (in both DNN)
  - Batch Size: 128
  - Replay Buffer: 1e5
  - Gamma: 0.99
  - Tau: 1e-3
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)

- For the neural models:    
  - Actor    
    - Hidden: (input, 256)  - ReLU
    - Hidden: (256, 128)    - ReLU
    - Output: (128, 4)      - TanH

  - Critic
    - Hidden: (input, 256)              - ReLU
    - Hidden: (256 + action_size, 128)  - ReLU
    - Output: (128, 1)                  - Linear
