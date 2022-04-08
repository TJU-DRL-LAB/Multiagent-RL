from __future__ import print_function

import json
import random

from .util import generate_uuid
from .dataset import Example
from .event import Event
from threading import Lock

class Controller(object):
    """
    Interface of the controller: takes two systems and run them to generate a dialgoue.
    """
    def __init__(self, scenario, sessions, chat_id=None, allow_cross_talk=False, session_names=(None, None)):
        self.lock = Lock()
        self.scenario = scenario
        self.sessions = sessions
        self.session_names = session_names
        self.chat_id = chat_id
        assert len(self.sessions) == 2
        self.events = []
        self.max_turns = None
        self.allow_cross_talk = allow_cross_talk
        self.session_status = {agent: 'received' for agent, _ in enumerate(self.sessions)}

        self.time_tmp = 0

    def describe_scenario(self):
        #print('='*50)
        for session in self.sessions:
            # print('\nAGENT={}'.format(session.agent))
            session.kb.dump()
            # int(session.kb)
        #print('='*50)
        return True

    def event_callback(self, event):
        raise NotImplementedError

    def get_outcome(self):
        raise NotImplementedError

    def get_result(self, agent_idx):
        return None

    def simulate(self, max_turns=None, verbose=False, temperature=1, eval=False):
        '''
        Simulate a dialogue.
        '''
        self.events = []
        self.max_turns = max_turns
        time = 0
        num_turns = 0
        game_over = False
        if random.random() < 0.5:
            first_speaker = 0
        else:
            first_speaker = 1
        print(self.scenario.kbs[0].facts)
        print(self.scenario.kbs[1].facts)
        if self.sessions[0].env.name == 'pt-neural-s':
            print("you are buyer")
        else:
            print("you are seller")
        while not game_over:
            for agent, session in enumerate(self.sessions):
                if num_turns == 0 and agent != first_speaker:
                    continue
                self.time_tmp = time
                event = session.send(temperature=temperature, eval=eval)
                print(event.data)
                time += 1
                if not event:
                    continue

                event.time = time
                # print('[{}]: {}'.format(agent, event.action))
                self.event_callback(event)
                self.events.append(event)

                # if verbose:
                #     event_tmp = event.to_dict()
                #     #event_tmp.pop('enc_output')
                #     event_tmp['metadata'] = None
                #     print('agent=%s: session=%s, event=%s' % (agent, type(session).__name__, event_tmp))
                # else:
                # if verbose:
                #     action = event.action
                #     data = event.data
                #     event_output = data if action == 'message' else "Action: {0}, Data: {1}".format(action, data)
                #     print('agent=%s, event=%s' % (agent, event_output))

                num_turns += 1
                if self.game_over() or (max_turns and num_turns >= max_turns):
                    game_over = True
                    # Let another one providing a quit action (for trainning value function)
                    partner = agent ^ 1
                    other_session = self.sessions[partner]

                    # for self u2 = rej/acpt: add u3 quit
                    session.receive_quit()

                    # for opponent u2 = offer: add u3 rej/acpt
                    other_session.receive(event)

                    break

                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)

        # Detect empty dialogues
        for s in self.sessions:
            if len(s.dialogue.lf_turns) == 0:
                print('error lfs: ', [s.dialogue.lf_turns for ss in self.sessions])
                print([str(e.action) + str(e.data) for e in self.events])
                print(self.outcomes, self.offers, self.quit)
                quit()

        uuid = generate_uuid('E')
        outcome = self.get_outcome()
        if verbose:
            # print('outcome: %s' % outcome)
            print('----------------')
        # TODO: add configurable names to systems and sessions
        agent_names = {'0': self.session_names[0], '1': self.session_names[1]}
        return Example(self.scenario, uuid, self.events, outcome, uuid, agent_names)


    def step(self, backend=None):
        '''
        Called by the web backend.
        '''
        with self.lock:
            # try to send messages from one session to the other(s)
            for agent, session in enumerate(self.sessions):
                if session is None:
                    # fail silently, this means that the session has been reset and the controller is effectively
                    # inactive
                    continue
                if (not self.allow_cross_talk) and self.session_status[agent] != 'received':
                    continue
                event = session.send()
                if event is None:
                    continue

                if not event.action in Event.decorative_events:
                    self.session_status[agent] = 'sent'
                self.event_callback(event)
                self.events.append(event)
                if backend is not None:
                    backend.add_event_to_db(self.get_chat_id(), event)

                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)
                        if not event.action in Event.decorative_events:
                            self.session_status[partner] = 'received'

    def inactive(self):
        """
        Return whether this controller is currently controlling an active chat session or not (by checking whether both
        users are still active or not)
        :return: True if the chat is active (if both sessions are not None), False otherwise
        """
        for s in self.sessions:
            if s is None:
                return True
        return False

    def set_inactive(self, agents=[]):
        """
        Set any number of sessions in the Controller to None to mark the Controller as inactive. The default behavior
        is to set all sessions to None (if no parameters are supplied to the function), but a list of indices can be
        passed to set the Session objects at those indices to None.
        :param agents: List of indices of Sessions to mark inactive. If this is None, the function is a no-op. If no
        list is passed, the function sets all Session objects to None.
        """
        with self.lock:
            if agents is None:
                return
            elif len(agents) == 0:
                self.sessions = [None] * len(self.sessions)
            else:
                for idx in agents:
                    self.sessions[idx] = None

    def get_chat_id(self):
        return self.chat_id

    def F(self):
        """Whether the game/session reaches the terminal state.
        """
        raise NotImplementedError

    def complete(self):
        """Whether the task was completed successfully.
        """
        raise NotImplementedError
