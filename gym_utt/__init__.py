import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='UTT-v0',
    entry_point='gym_utt.envs:UTTEnv',
)
