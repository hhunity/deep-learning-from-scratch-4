import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self):
        #行動パターン
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        #報酬マップ
        self.reward_map = np.array(
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        # ゴール（りんご）
        self.goal_state = (0, 3)
        # 壁
        self.wall_state = (1, 1)
        # エージェントの初期位置
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    #行動にアクセスするメソッド
    def actions(self):
        return self.action_space
    #状態にアクセスするメソッド
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)
    #環境の状態遷移を表すメソッド
    def next_state(self, state, action):
        #移動先の場所の計算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        #グリッドのワールド外か？壁か？
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        #次の状態を返す
        return next_state

    #報酬関数メソッド
    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    #ゲームを初期状態にする
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state
    #行動する
    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    #可視化のメソッド
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)
