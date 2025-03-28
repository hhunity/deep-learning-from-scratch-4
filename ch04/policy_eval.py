if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld

#反復方策の実装
def eval_onestep(pi:defaultdict, V:defaultdict, env:GridWorld, gamma:float=0.9):
    """
        pi(defaultdic):方策
        v(defaultdic):価値関数
        env(GridWOrld):環境
        gamma(float):割引率
    """
    for state in env.states(): #各状態(各グリッド）へのアクセス
        if state == env.goal_state: #ゴールの価値関数は常に0
            V[state] = 0
            continue
        
        #各行動の確率
        action_probs = pi[state]
        new_V = 0
        #各行動へアクセス
        #actition_probsは辞書なので、actionには０〜４、action_probは確率が入る
        for action, action_prob in action_probs.items():
            #次の状態
            next_state = env.next_state(state, action)
            #報酬
            r = env.reward(state, action, next_state)
            #新しい価値関数
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()#更新前の価値関数

        V = eval_onestep(pi, V, env, gamma)

        #更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        #閾値の比較(更新された量の最大値が閾値未満なら終了)
        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)

