from src.scenarios.reward_enum import RewardType
from vmas.simulator.scenario import BaseScenario
from vmas.scenarios.balance import Scenario as DefaultBalanceScenario
from vmas.scenarios.buzz_wire import Scenario as DefaultBuzzWireScenario
from vmas.scenarios.football import Scenario as DefaultFootballScenario
from vmas.scenarios.transport import Scenario as DefaultTransportScenario
from vmas.scenarios.navigation import Scenario as DefaultNavigationScenario
from vmas.scenarios.wheel import Scenario as DefaultWheelScenario
from src.scenarios.balance.wrappers.const_reward import ConstRewardBalanceScenario
# from src.scenarios.balance.wrappers.model_reward import ModelRewardBalanceScenario
# from src.scenarios.football.wrappers.model_reward import ModelRewardFootballScenario
# from src.scenarios.buzz_wire.wrappers.model_reward import ModelRewardBuzzWireScenario
# from src.scenarios.transport.wrapper.model_reward import ModelRewardTransportScenario
# from src.scenarios.navigation.wrappers.model_reward import ModelRewardNavigationScenario
# from src.scenarios.wheel.wrapper.model_reward import ModelRewardWheelScenario

class WrapperFactory:
    """
    A Factory that produces a wrapper instance given an input of enum BalanceReward
    Simple Instance of the Factory Design Pattern (see https://www.geeksforgeeks.org/factory-method-python-design-patterns/)
    """
    @staticmethod
    def create(scenario: str, wrapper: RewardType, **kwargs) -> BaseScenario:
        if wrapper is None:
            wrapper = RewardType.NO_WRAP

        # If the user indicated that they want to use an RM, make sure they passed "model"
        if wrapper.value == RewardType.LEARNED_REWARD.value:
            if kwargs.get("model", None) is None:
                raise Exception(
                    "Tried to Instantiate a Reward Wrapper with a Torch model, but none was provided to the factory")

        # Switch Conditions for returning the correct scenario
        # Balance Wrappers
        if scenario == "balance":
            if wrapper.value == RewardType.NO_WRAP.value:
                return DefaultBalanceScenario()
            elif wrapper.value == RewardType.CONSTANT.value:
                return ConstRewardBalanceScenario()
            # elif wrapper.value == RewardType.LEARNED_REWARD.value:
            #     model = kwargs.get("model", None)
            #     return ModelRewardBalanceScenario(model)

        # Buzz_Wire Wrappers
        elif scenario == "buzz_wire":
            if wrapper.value == RewardType.NO_WRAP.value:
                return DefaultBuzzWireScenario()
            # elif wrapper.value == RewardType.LEARNED_REWARD.value:
            #     model = kwargs.get("model", None)
            #     return ModelRewardBuzzWireScenario(model)

        # Football Wrappers
        elif scenario == "football":
            if wrapper.value == RewardType.NO_WRAP.value:
                return DefaultFootballScenario()
            # elif wrapper.value == RewardType.LEARNED_REWARD.value:
            #     model = kwargs.get("model", None)
            #     return ModelRewardFootballScenario(model)

        # Transport Wrappers
        elif scenario == "transport":
            if wrapper.value == RewardType.NO_WRAP.value:
                return DefaultTransportScenario()
            # elif wrapper.value == RewardType.LEARNED_REWARD.value:
            #     model = kwargs.get("model", None)
            #     return ModelRewardTransportScenario(model)

        # Navigation Wrappers
        elif scenario == "navigation":
            if wrapper.value == RewardType.NO_WRAP.value:
                return DefaultNavigationScenario()
            # elif wrapper.value == RewardType.LEARNED_REWARD.value:
            #     model = kwargs.get("model", None)
            #     return ModelRewardNavigationScenario(model)

        elif scenario == "wheel":
            if wrapper.value == RewardType.NO_WRAP.value:
                return DefaultWheelScenario()
            # elif wrapper.value == RewardType.LEARNED_REWARD.value:
            #     model = kwargs.get("model", None)
            #     return ModelRewardWheelScenario(model)

        raise Exception(f"Could not create wrapper with type {wrapper} for scenario {scenario}, please use only the supported enum values of RewardType and scenarios")