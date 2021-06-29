# footballlearning
The main topic of this project was to teach an autonomous robot how to play a table football game with another human-controlled robot,
using reinforcement learning algorithm called Q-Learning.
The code was written in Python and took place in a robotic simulation called CoppeliaSim (V-rep).

All the data is inserted in the codes of Policy_Game and q_learning_first_run (the CoppeliaSim file location, the values of the parameters and etc).
In order to start the programs all you need to do is to press on the Play button on the top and make sure that you are choosing 
the correct python file.



In q_learning_first_runs you need each time to update the Q_table that you are reading from and the Q_table that you are writing to.
In addition, you need to update few parameters: exploration_rate, episodes - each run you need to add +500 to the latest.


In order to see the performance of the robot after it finished the learning session, you need to read from Q_table27, as 
this is the last file of the learning.
