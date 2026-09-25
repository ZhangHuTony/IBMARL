from .balance.wrappers.no_early_stopping import NoEarlyStopBalanceScenario as BalanceScenario
from vmas.scenarios.reverse_transport import Scenario as ReverseTransportScenario
from vmas.scenarios.football import Scenario as FootballScenario
from .navigation.wrappers.no_early_stopping import NoEarlyStopNavigationScenario as NavigationScenario
from vmas.scenarios.buzz_wire import Scenario as BuzzWireScenario
from vmas.scenarios.transport import Scenario as TransportScenario
from .diffdrive_navigation.scenario import DiffDriveNavigation
from vmas.scenarios.wheel import Scenario as WheelScenario
from .diffdrive_pusher.scenario import Scenario as DiffDrivePusherScenario

class ScenarioFactory:
    """
    Factory Class
    Given the name of an environment, return the appropriate scenario from VMAS
    This allows for string-parameterized instantiation of scenarios for demonstrations
    """
    @staticmethod
    def create(env_name, **kwargs):
        if env_name == 'balance':
            return BalanceScenario()
        elif env_name == 'reverse_transport':
            return ReverseTransportScenario()
        elif env_name == "transport":
            return TransportScenario()
        elif env_name == "football":
            return FootballScenario()
        elif env_name == "navigation":
            return NavigationScenario()
        elif env_name == "buzz_wire":
            return BuzzWireScenario()
        elif env_name == "wheel":
            return WheelScenario()
        elif env_name == "diffdrive_navigation":
            return DiffDriveNavigation()
        elif env_name == "diffdrive_pusher":
            return DiffDrivePusherScenario()
        raise ValueError(f'Scenario Factory: Unknown scenario env name: {env_name}')
