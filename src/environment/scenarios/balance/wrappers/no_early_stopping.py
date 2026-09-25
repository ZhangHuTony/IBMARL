from vmas.scenarios.balance import Scenario as BalanceScenario

class NoEarlyStopBalanceScenario(BalanceScenario):

    def __init__(self):
        """
        Create the Scenario
        :param kwargs: Assumes the same set of arguments as vmas.scenarios.balance.BaseScenario.__init__
        """
        super().__init__()

    def done(self):
        """
        Set to False to prevent the simulation from ending early.
        """
        done = super().done()
        done.fill_(False)
        return done
