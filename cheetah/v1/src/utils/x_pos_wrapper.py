import gymnasium as gym


class XPosWrapper(gym.Wrapper):
    """
    Wraps MuJoCo envs to inject info['x_pos'] using sim qpos[0].
    Falls back to obs[0] if access fails.
    """

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        x = self._get_x_pos_fallback(obs)
        info = dict(info)
        info["x_pos"] = x
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        x = self._get_x_pos_fallback(obs)
        info = dict(info)
        info["x_pos"] = x
        return obs, info

    def _get_x_pos_fallback(self, obs):
        try:
            return float(self.env.unwrapped.data.qpos[0])
        except Exception:
            return float(obs[0])