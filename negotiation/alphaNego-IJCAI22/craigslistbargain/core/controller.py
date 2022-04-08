from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None, session_names=(None, None)):
        super(Controller, self).__init__(scenario, sessions, chat_id, session_names=session_names)
        # self.prices = [None, None]
        self.offers = [None, None]
        # self.sides = [None, None]
        self.outcomes = [None, None]
        self.quit = False

        self._tom_hidden = None

    def fake_step(self, agent, event, tom_session=None):
        '''
               Simulate a dialogue.
        '''
        session = self.sessions[agent]
        # event = session.send()
        time = self.time_tmp
        if not event:
            print('Error: Event')
            return
        event.time = time + 1

        for partner, other_session in enumerate(self.sessions):
            if agent != partner:
                if tom_session is not None and not isinstance(tom_session, bool):
                    self._tom_hidden = tom_session.tom_hidden
                    tom_session.receive(event)
                    info_back = tom_session.send(is_fake=True, strategy=other_session.price_strategy_label)
                    return info_back
                else:
                    other_session.receive(event)
                    info_back = other_session.send(is_fake=True)
                    return info_back

    def get_value(self, agent, events):
        for partner, other_session in enumerate(self.sessions):
            if agent != partner:
                return other_session.get_value(events)

    def step_back(self, agent, tom_session):
        if not isinstance(tom_session, bool):
            tom_session.tom_hidden = self._tom_hidden
            tom_session.step_back()
        else:
            for partner, other_session in enumerate(self.sessions):
                if agent != partner:
                    other_session.step_back()

    def event_callback(self, event):
        if event.action == 'offer':
            self.offers[event.agent] = event.metadata
        elif event.action == 'accept':
            self.outcomes[event.agent] = True
        elif event.action == 'reject':
            self.outcomes[event.agent] = False
        elif event.action == 'quit':
            self.quit = True
            self.outcomes[event.agent] = False

    def get_margin_reward(self, price, agent, is_agreed=True):
        # No agreement
        if not is_agreed:
            return -0.5

        if price is None:
            if self.offers[0] is not None:
                price = self.offers[0]['price']
            elif self.offers[1] is not None:
                price = self.offers[1]['price']
            else:
                print('Incorrect tom')
                raise NotImplementedError()

        rewards = {}
        targets = {}
        kbs = [self.sessions[i].kb for i in range(2)]
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']
        return rewards[self.sessions[agent].kb.role]

    def get_outcome(self):
        offer = None
        reward = 0
        if self.offers[0] is not None and self.outcomes[1] is True:
            reward = 1
            offer = self.offers[0]
        elif self.offers[1] is not None and self.outcomes[0] is True:
            reward = 1
            offer = self.offers[1]
        else:
            if (self.offers[0] is not None or self.offers[1] is not None) and False in self.outcomes:
                reward = 0
                offer = self.offers[0] if self.offers[1] is None else self.offers[1]

        # possible outcomes:
        # reward is 1 and offer is not null: complete dialogue
        # reward is 0 and offer is not null: incomplete dialogue (disagreement): offer was made and not accepted
        # reweard is 0 and offer is null: incomplete dialogue: no offer was made
        return {'reward': reward, 'offer': offer}

    def game_over(self):
        return not self.inactive() and \
               ((self.offers[0] is not None and self.outcomes[1] is not None) or
                (self.offers[1] is not None and self.outcomes[0] is not None) or
                 self.quit)

    def get_result(self, agent_idx):
        # todo fix this if we ever want to display results in the survey
        return None

    def complete(self):
        return (self.offers[0] is not None and self.outcomes[1] is True) or (self.offers[1] is not None and self.outcomes[0] is True)

    def get_winner(self):
        # todo fix this if we ever want to calculate who the winner is
        return -1
