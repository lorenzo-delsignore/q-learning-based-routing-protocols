# Q-Learning-based-routing-protocol

# Introduction
Let us consider N drones ${d_0, ..., d_{N−1}}$ deployed in an Area of Interest (AoI). Each drone $d_i$ has a
mission assigned which consists of following a trajectory in the AoI and capturing events. Such events
generate packets that have to be sent to the depot. Once a packet has been generated, the drone, can
either keep it in its buffer until it reaches the depot or finds a drone to use as a relay and transmit
all the packets in its buffer to the relay. Packets can also expire and have a deadline to be delivered,
therefore our goal is to deliver them as quickly as possible to the depot.

# Approaches

To solve the routing protocol the Q-Learning algorithm was used which is an off-policy TD control algorithm in Reinforcement Learning:
 - greedy (with exploration in the early stages)
 - greedy (without exploratin in the early stages)
 - best action (with exploration in the early stages)
 - best action (without exploration in the early stages)
 - Q-FANET 
 
The simulator used for the experiments can be found at this link https://github.com/flaat/DroNETworkSimulator/

If you want to try the solutions, you can put the routing algorithms in this folder https://github.com/flaat/DroNETworkSimulator/tree/main/src/routing_algorithms
 
