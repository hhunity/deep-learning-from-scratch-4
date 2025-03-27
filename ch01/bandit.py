import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """
            np.random.rand()は浮動小数点で０〜１の間の乱数
            np.random.randint(low, high, size)はlow,highまでの整数の乱数。
            サイズを指定したら多次元も作れる。
        
        """
        if np.random.rand() < self.epsilon:
            #np.random.randint(low, high=None, size=None, dtype=int)
            #lowからhaighまでの乱数
            #high を指定しない場合、0 から low の範囲で整数が生成される。
            return np.random.randint(0, len(self.Qs)) #ランダムにマシンを選ぶ
        return np.argmax(self.Qs) #期待値が最大のインデックスを返す


if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action() #行動を選ぶ
        reward = bandit.play(action)#実際にプレイして報酬を得る
        agent.update(action, reward)#行動と報酬から学ぶ
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print(total_reward)

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()
