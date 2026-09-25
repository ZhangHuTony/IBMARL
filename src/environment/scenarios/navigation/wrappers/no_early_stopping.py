from vmas.scenarios.navigation import Scenario as NavigationScenario

class NoEarlyStopNavigationScenario(NavigationScenario):

    def __init__(self, **kwargs):
        """
        Create the Scenario
        :param kwargs: Assumes the same set of arguments as vmas.scenarios.balance.BaseScenario.__init__
        """
        super().__init__(**kwargs)

    def done(self):
        """
        Set to False to prevent the simulation from ending early.
        """
        done = super().done()
        done.fill_(False)
        return done
