import numpy as np

from src.entities.uav_entities import Packet, HelloPacket, ACKPacket, DataPacket
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util, config
from collections import defaultdict, deque


class QFANET(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)

        self.outcome_reward_mapper = {-1: -100, 1: 100, 0: 50}

        self.sinr_weight = 0.7
        self.epsilon = 0.1  # 0.10 funziona molto bene con best_action
        self.alpha = 0.6
        self.look_back = 10  # the max episodes to consider for the old rewards

        self.taken_actions = {}  # id event : (old_state, old_action)
        self.q_table = defaultdict(lambda: 0.5)  # { action: qvalue }
        self.reward_table = defaultdict(deque)  # action: [old_rewards]

        # store neighbour delivery to be used in updating Q-Value
        # we only update delay when nodes have been selected as relay for a successful communication (ACKed)
        self.neighbours_delay = defaultdict()

        self.log = defaultdict(lambda: 0)  # for logging

    def update_qvalue(self, action_id, reward, channel_success=0):
        # update Q-Value => (old rewards * their weights) + current reward + channel_success
        # the more recent a reward is, the more weight it has in Q-Value
        normalizer = sum(range(1, len(self.reward_table[action_id]) + 1))
        limited_q_value = 0
        for index, value in enumerate(self.reward_table[action_id]):
            normalized_weight = (index + 1) / normalizer  # dynamic weight of the old reward
            limited_q_value += normalized_weight * value

        self.q_table[action_id] = (1 - self.alpha) * limited_q_value + (
                self.alpha * reward) + (self.sinr_weight * channel_success)

    def feedback(self, drone, id_event, delay, outcome):
        """
        Feedback returned when the packet arrives at the depot or
        Expire. This function have to be implemented in RL-based protocols ONLY
        @param drone: The drone that holds the packet
        @param id_event: The Event id
        @param delay: packet delay
        @param outcome: -1 or 1 or 0 (read below)
        @return:
        """

        # outcome can be:
        #  -1 if the packet/event expired;
        #   1 if the packets has been delivered to the depot
        #   0 if the drone "drone" was selected as relay - handled `drone_reception`

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple
        # feedback for the same packet!!

        # update drone delay to be used in subsequent routing decisions
        if outcome == 0:
            self.neighbours_delay[drone.identifier] = delay  # update delivery delay for this neighbour

        # calculate the channel success rate between drones, (will be used to update Q-Value)
        distance = util.euclidean_distance(self.drone.coords, drone.coords)
        channel_success = self.gaussian_success_handler(distance) if outcome == 0 else 0

        if id_event in self.taken_actions:
            # BE AWARE, IMPLEMENT YOUR CODE WITHIN THIS IF CONDITION OTHERWISE IT WON'T WORK!
            # TIPS: implement here the q-table updating process
            action = self.taken_actions[id_event]
            action_id = action.identifier if action is not None else self.drone.identifier

            # remove the entry, the action has received the feedback
            del self.taken_actions[id_event]

            # calculate reward for the current (selected) action
            reward = self.outcome_reward_mapper.get(outcome, 0)

            # update q-value
            self.update_qvalue(action_id, reward, channel_success)

            # keep track of old rewards for the current (selected) action
            # NOTE: Make sure to calculate Q-VALUE before appending current reward to the old rewards
            self.reward_table[action_id].append(reward)

            # memory optimization, we don't need to keep track of rewards, more than the lookback value
            if len(self.reward_table[action_id]) > self.look_back:
                self.reward_table[action_id].popleft()

    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.

        if candidates' list is empty then use QMR

            QMR: if there are neighbours with velocity more than 0, then select the max. one
                 else apply penalty mechanism

        else Use Q-Noise+ (Q-Learning based approach)

        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @param packet:
        @return: The best drone to use as relay
        """
        # calculate dynamic velocity constraint
        depot_pos = self.drone.depot.coords
        drone_pos = self.drone.coords
        delivery_deadline = self.simulator.event_duration
        velocity_constraint = util.euclidean_distance(depot_pos, drone_pos) / delivery_deadline

        # select candidate neighbors who satisfy velocity constraint
        candidates = self.filter_velocity_constraint(opt_neighbors, velocity_constraint)

        if len(candidates) == 0:
            relay = self.get_relay_by_qmr(opt_neighbors)
        else:
            relay = self.get_relay_by_qnoise(candidates)

        # store the taken action
        self.taken_actions[packet.event_ref.identifier] = relay if relay is not None else self.drone

        return relay

    def get_relay_by_qmr(self, opt_neighbors: list):
        velocity_constraint = 0
        candidates = self.filter_velocity_constraint(opt_neighbors, velocity_constraint)

        if len(candidates) == 0:  # this will automatically trigger penalty mechanism when the packet expire
            self.log['qmr_empty'] += 1
            return None

        self.log['qmr'] += 1
        max_velocity = 0
        best_relay = None
        for velocity, candidate in candidates:
            if velocity > max_velocity:
                max_velocity = velocity
                best_relay = candidate

        return best_relay

    def get_relay_by_qnoise(self, candidates: list):
        """
        This function returns the best relay to send packets (derived by argmax of Q table).
        """
        relay = None

        # do exploration if force exploration, or in the epsilon case
        if self.simulator.rnd_routing.rand() < self.epsilon:
            # select a random neighbor
            relays = [drone[1] for drone in candidates]
            relay = self.simulator.rnd_routing.choice(relays)
        else:
            # select the best by argmax
            best_qvalue = None
            for _, neighbor in candidates:
                qvalue = self.q_table[neighbor.identifier]
                if best_qvalue is None or qvalue > best_qvalue:
                    best_qvalue = qvalue
                    relay = neighbor
        self.log['q-noise'] += 1
        return relay

    def filter_velocity_constraint(self, opt_neighbors: list, velocity_constraint=0):
        """ Calculate the list of neighbors who satisfy the velocity constraint """
        depot_pos = self.drone.depot.coords
        drone_pos = self.drone.coords
        candidates = []
        for hello_pkt, neighbor in opt_neighbors:
            neighbor_pos = hello_pkt.cur_pos
            drone_2_depot = util.euclidean_distance(depot_pos, drone_pos)
            neighbor_2_depot = util.euclidean_distance(depot_pos, neighbor_pos)

            delay = self.neighbours_delay.get(neighbor.identifier, 0)  # delay between nodes i & j
            if delay == 0:  # can happen on the start when delay table is empty
                delay = hello_pkt.time_delivery - hello_pkt.time_step_creation

            velocity = (drone_2_depot - neighbor_2_depot) / delay
            if velocity > velocity_constraint:
                candidates.append((velocity, neighbor))
        return candidates

    def drone_reception(self, src_drone, packet: Packet, current_ts):
        """
            We override this function to
            1.  Add arrival time for hello messages
            2.  Handle Forward feedback
        """
        super().drone_reception(src_drone, packet, current_ts)

        if isinstance(packet, HelloPacket):
            packet.time_delivery = current_ts
        elif isinstance(packet, DataPacket):
            # Add reception time to the packet
            # this will propagate back to the src_drone by AckPacket for delay calculation and updating delay ta
            #
            # Method 1: this use overall end-to-end delay
            packet.time_delivery = current_ts

            # Method 2: This method will use delay between two nodes.
            if packet.optional_data is None:
                packet.optional_data = deque([packet.time_step_creation], maxlen=2)
            packet.optional_data.append(current_ts)
        elif isinstance(packet, ACKPacket):
            # Method 1: this use overall end-to-end delay
            delivery_delay = packet.acked_packet.time_delivery - packet.acked_packet.time_step_creation

            # Method 2: This method will use delay between two nodes.
            # reception_times = packet.acked_packet.optional_data
            # delivery_delay = reception_times[1] - reception_times[0]

            src_drone.routing_algorithm.feedback(self.drone, packet.event_ref.identifier, delivery_delay, 0)
