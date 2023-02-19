from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
import numpy as np
from collections import defaultdict

class QLearningRouting_GreedyNoForce(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)

        self.taken_actions = {}  # id event : (old_state, old_action)
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0)) # cell: { action: qvalue }
        self.epsilon = 0.10 # 0.10 funziona molto bene con best_action
        self.alpha = 0.20
        self.gamma = 0.85

    def feedback(self, drone, id_event, delay, outcome):
        """
        Feedback returned when the packet arrives at the depot or
        Expire. This function have to be implemented in RL-based protocols ONLY
        @param drone: The drone that holds the packet
        @param id_event: The Event id
        @param delay: packet delay
        @param outcome: -1 or 1 (read below)
        @return:
        """

        # outcome can be:
        #   -1 if the packet/event expired;
        #   1 if the packets has been delivered to the depot

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple
        # feedback for the same packet!!

        if id_event in self.taken_actions:
            # Drone id and Taken actions
            # print(f"\nIdentifier: {self.drone.identifier}, Taken Actions: {self.taken_actions}, Time Step: {self.simulator.cur_step}")
            action, state, next_state = self.taken_actions[id_event]
            action_identifier = action.identifier if action != drone else self.drone.identifier

            # remove the entry, the action has received the feedback
            del self.taken_actions[id_event]

            if outcome == -1:
                reward = -1
            elif outcome == 1:
                # higher reward on lower delay
                reward = (self.simulator.event_duration - delay) / self.simulator.event_duration

            # reward or update using the old state and the selected action at that time
            q_state = self.q_table[state][action_identifier]
            future_reward = 0 if len(self.q_table[next_state].keys()) == 0 else max(self.q_table[next_state].values())
            self.q_table[state][action_identifier] = q_state + self.alpha * (reward + self.gamma * future_reward - q_state)

    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.

        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """
        
        # get the current state and the next state of the current agent
        state = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell, width_area=self.simulator.env_width, x_pos=self.drone.coords[0], y_pos=self.drone.coords[1])[0]
        next_state = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell, width_area=self.simulator.env_width, x_pos=self.drone.next_target()[0], y_pos=self.drone.next_target()[1])[0]
        
        # get best relay to send packets
        policy_name = "greedy_policy" # greedy_policy | best_action_policy
        action = self.get_action(state, opt_neighbors, policy_name=policy_name)

        # store the taken action
        self.taken_actions[packet.event_ref.identifier] = (action, state, next_state)

        # keep the packet
        if action == self.drone:
            return None

        return action

    def get_action(self, state, opt_neighbors, policy_name="best_action_policy"):
        """
        Get the action random with probability epsilon, greedily otherwise.

        @param state: the state from where to take the decision.
        @return: the greedy or random action.
        """

        assert policy_name in [ "greedy_policy", "best_action_policy" ]
    
        # do exploration if force exploration, or in the epsilon case
        if self.simulator.rnd_routing.rand() < self.epsilon:
            # keep the packet
            if len(opt_neighbors) == 0:
                return None

            # select a random neighbor
            relays = [ drone[1] for drone in opt_neighbors ]
            return self.simulator.rnd_routing.choice(relays)
        
        # do exploitation, using a well-defined policy for relay selection
        policy = getattr(self, policy_name)
        return policy(state, opt_neighbors)

    def best_action_policy(self, state, opt_neighbors):
        relay = None
        best_score = None
        me_depot_distance = self.drone.distance_from_depot

        for _, neighbor in opt_neighbors:
            drone_depot_distance = neighbor.distance_from_depot
            
            # since the current drone is closer to the depot than the neighbor, keep the last best relay as the best one
            if me_depot_distance < drone_depot_distance:
                continue

            # the neighbor is in the depot range, get this neighbor as the best relay
            if drone_depot_distance <= self.simulator.depot_com_range:
                relay = neighbor
                break

            # calculate score based by the depot distance (the lower the drone from the depot the better)
            qvalue = self.q_table[state][neighbor.identifier]
            score = 1 / drone_depot_distance * qvalue

            if best_score is None or score > best_score:
                best_score = qvalue
                relay = neighbor

        return relay if relay != None else self.drone

    def greedy_policy(self, state, opt_neighbors):
        """
        This function returns the best relay to send packets (derived by argmax of Q table).
        """

        relay = None
        best_qvalue = None

        for _, neighbor in opt_neighbors:
            qvalue = self.q_table[state][neighbor.identifier]

            # the neighbor is in the depot range, get this neighbor as the best relay
            if neighbor.distance_from_depot <= self.simulator.depot_com_range:
                return neighbor

            # select the best by argmax
            if best_qvalue is None or qvalue > best_qvalue:
                best_qvalue = qvalue
                relay = neighbor

        return relay if relay != None else self.drone
