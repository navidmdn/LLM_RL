
from envs.gsm8k import GSM8kEnv
from algorithms.reinforce import Reinforce


def run():

    env_config = {}
    agent_config = {}

    env = GSM8kEnv(**env_config)
    agent = Reinforce(env, **agent_config)
    agent.train()


