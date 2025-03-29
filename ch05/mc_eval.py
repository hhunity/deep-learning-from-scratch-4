import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        #方策
        #piは位置（タプル）を引数に取る
        self.pi = defaultdict(lambda: random_actions)
        #価値関数
        self.V = defaultdict(lambda: 0)
        #報酬の算出に使う
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        """
            stateにおける行動を1つ取り出す
        """
        #状態（場所）の確率分布を取得
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        #行動の確率に従って1つ選択するpが確率分布、それに従い行動を選択する
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """
            行動を記録するためのメソッド
            memoryに書く行動ごとにタプルで詰める
        """
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        """
            モンテカルロ法を行うメソッド
            逆向きにたどりながら収益を計算していく
        """
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


env = GridWorld()
agent = RandomAgent()

#エピソード数
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    #エピソードごとにサンプルする
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.eval()
            break

        state = next_state

env.render_v(agent.V)