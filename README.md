# bio-inspired-2025-ae4350


##Setup instructions##

#Create and activate conda enfironment

conda create -n lunarlander-rl python=3.10
conda activate lunarlander-rl

#To install the packages, 
pip install -r requirements.txt



# information of the environment

state vector of lunar lander: 
0: x position
1: y position
2: x velocity
3: y velocity
4: angle of lander (radians)
5: angular velocity
6: left leg contact (0 or 1)
7: right leg contact (0 or 1)

actions of the lunar lander: 
0: nothing
1: fire left orientation engine
2: fire main (downward) engine
3: fire right orientation engine 


# code functioning

To evaluate the tuning of the different parameters, the order on which the parameters have been tuned are: 

First a nominal_baseline is run based on information from literature, to obtain directly the values that give approximately good results. Once this baseline is done and it has been seen to work correctly, a list of hyperparameters is evaluated to select the best solutions. They will be evaluated secuentially. Once one is evalauted, the best parameter will be selected and the following parameter will be then evaluated

- layers
- batch size
- learning rate
- epsilon decay
- epsilon min
- gamma

In the repository there are different codes, which serve for different purposes for the training of the network for the accomplishment of the lunar lander mission. 

- train_DQN.py: this code includes all the necessary functions to train the network. 
- tune_DQN.py: This code is done to tune one by one all the parameters selected. The code will call the trainAgent function from the train_DQN.py to train the agent. The data of the episodes will be saved in a csv file in tune/ under the name of the hyperparameter that is being tuned, and the value of the hyperparameter. Once the agent has reached convergence, or the number of episodes has reached it's maximum, the model is saved under the "saved_models/", this serves to make the trained agent play the game. The file is saved under the name of the name of the hyperparameter tuned, followed by the value of the hyperparameter together with the epusode number on which it converged. 
- plot_tuned.py: this code serves to plot the results of the different tuning. It access the selection of the hyperparameters list and it pltos the .csv files for comparison. 
- run_DQN.py: This function makes the agent play the lunar lander game, it is necessary to enter manually the name of the trained agent that wants to be run, on the saved/models folder. 

The flags need to be set as true to select the parameters that wants to be tuned. It has to be taken into consideration that if the code is run with any of the flags set as true, the csv file saved in tune will be overwritten with the new episodes. 