if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy


def value_iter_onestep(V:defaultdict, env:GridWorld, gamma:float):
    """
        V:状態ごとの価値関数
        env:環境
        gamma:割引率
    
    """
    #すべての状態（場所）でループ
    for state in env.states():
        #ゴールは価値が０
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_values = []
        #すべての行動にアクセス
        for action in env.actions():
            #次の状態
            next_state = env.next_state(state, action)
            #次の状態で得られる報酬
            r = env.reward(state, action, next_state)
            #価値関数を更新
            value = r + gamma * V[next_state]
            action_values.append(value)

        #すべての行動のうち、最大の価値が価値関数
        V[state] = max(action_values)
    return V


def value_iter(V:defaultdict, env:GridWorld, gamma:float, threshold:float=0.001, is_render:bool=True):
    """
        V:状態ごとの価値関数
        env:環境
        gamma:割引率
        threshold:方針評価を止める閾値
        is_render:方策の評価と改善を行う過程を描画するか
    """
    while True:
        if is_render:
            env.render_v(V)

        #更新前の価値関数
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        #更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        #閾値との比較
        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
