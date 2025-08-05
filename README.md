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
