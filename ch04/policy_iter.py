if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_eval import policy_eval

#辞書から、最大となるキーを求める
def argmax(d):
    """d (dict)"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

#価値関数をgreedy化する関数
def greedy_policy(V:defaultdict, env:GridWorld, gamma:float):
    """
        V:状態ごとの価値関数
        env:環境
        gamma:割引率
    
    """
    pi = {}

    #すべての状態（場所）でループ
    for state in env.states():
        action_values = {}

        #行動毎の行動価値を算出
        #すべての行動でループ
        for action in env.actions():
            #次の行動
            next_state = env.next_state(state, action)
            #次の行動における評価
            r = env.reward(state, action, next_state)
            #式(4.8)の更新部分
            value = r + gamma * V[next_state]
            #行動価値を格納
            action_values[action] = value

        #最大の価値関数を持つ行動を取り出す
        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        #最適行動が確率１となるように行動の確率分布更新する
        pi[state] = action_probs
    return pi


def policy_iter(env:GridWorld, gamma:float, threshold:float=0.001, is_render:bool=True):
    """
        env:環境
        gamma:割引率
        threshold:方針評価を止める閾値
        is_render:方策の評価と改善を行う過程を描画するか
    """
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        #評価
        V = policy_eval(pi, V, env, gamma, threshold)
        #改善
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        #更新確認
        if new_pi == pi:
            break
        pi = new_pi

    return pi


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
