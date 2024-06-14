import gym
import pybullet as p
import pybullet_data
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import matplotlib.pyplot as plt
import os

# PPO 하이퍼파라미터 설정
gamma = 0.99
lambda_gae = 0.95
epsilon = 0.2
learning_rate_actor = 0.0003
learning_rate_critic = 0.001
epochs = 10
batch_size = 64
memory_size = 2048
episodes = 1000

# Actor (정책 네트워크) 정의
def build_actor(state_shape, action_space):
    model = tf.keras.models.Sequential([
        layers.Input(shape=state_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_space, activation='softmax')
    ])
    return model

# Critic (가치 네트워크) 정의
def build_critic(state_shape):
    model = tf.keras.models.Sequential([
        layers.Input(shape=state_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model

# 경험 재생 버퍼
class Memory:
    def __init__(self):
        self.states = deque(maxlen=memory_size)
        self.actions = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.next_states = deque(maxlen=memory_size)
        self.dones = deque(maxlen=memory_size)
        self.log_probs = deque(maxlen=memory_size)

    def store(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()

# 어드밴티지 및 리턴 계산
def compute_advantages(rewards, values, next_values, dones, gamma, lambda_gae):
    advantages = []
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lambda_gae * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
    return np.array(advantages), np.array(returns)

# 행동 선택
def choose_action(state, actor):
    state = state.reshape([1, *state.shape])
    action_probs = actor.predict(state, verbose=0).flatten()
    action = np.random.choice(len(action_probs), p=action_probs)
    log_prob = np.log(action_probs[action])
    return action, log_prob

# 학습 과정
def train(memory, actor, critic, actor_optimizer, critic_optimizer, batch_size, epochs, gamma, lambda_gae, epsilon):
    states = np.array(memory.states)
    actions = np.array(memory.actions)
    rewards = np.array(memory.rewards)
    next_states = np.array(memory.next_states)
    dones = np.array(memory.dones)
    old_log_probs = np.array(memory.log_probs)

    values = critic.predict(states, verbose=0)
    next_values = critic.predict(next_states, verbose=0)
    advantages, returns = compute_advantages(rewards, values, next_values, dones, gamma, lambda_gae)

    for epoch in range(epochs):
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        for i in range(0, len(states), batch_size):
            idx = indices[i:i + batch_size]
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]
            batch_old_log_probs = old_log_probs[idx]

            with tf.GradientTape() as tape:
                action_probs = actor(batch_states, training=True)
                action_log_probs = tf.math.log(tf.reduce_sum(action_probs * tf.one_hot(batch_actions, actor.output_shape[-1]), axis=1))
                ratios = tf.exp(action_log_probs - batch_old_log_probs)
                clipped_ratios = tf.clip_by_value(ratios, 1 - epsilon, 1 + epsilon)
                actor_loss = -tf.reduce_mean(tf.minimum(ratios * batch_advantages, clipped_ratios * batch_advantages))

            actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

            critic_loss = critic.train_on_batch(batch_states, batch_returns)

    memory.clear()

# 환경 설정
class EndoscopeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(6)  # 6방향 이동: 앞, 뒤, 좌, 우, 위, 아래
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)  # 상태 변수에 벽과의 거리 추가
        self.client = p.connect(p.DIRECT)  # DIRECT 모드로 변경하여 리소스 절약
        self.reset()

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.maze_size = 3  # X, Y 축 길이
        self.wall_height = 3  # Z 축 길이 (미로의 높이를 확장)
        self.wall_thickness = 0.1
        self.agent_start_pos = [0.5, 0.5, 0.5]  # 초기 위치 고정
        self.goal_pos = [2.5, 2.5, 2.5]  # 목표 위치를 Z축 방향으로 높임
        self.goal_size = 0.5  # 목표 영역의 반경 (1x1x1 큐브의 중심)
        self.walls = [
            [(1, 0, 0), (1, 2, 0)],
            [(2, 1, 0), (2, 3, 0)],
            [(1, 0, 0), (1, 0, 2)],
            [(2, 1, 0), (2, 1, 3)]
        ]
        self.create_borders()  # 테두리 벽 생성
        for wall in self.walls:
            self.create_wall(wall[0], wall[1])
        self.agent = p.loadURDF("r2d2.urdf", self.agent_start_pos, physicsClientId=self.client)
        self.goal_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.goal_size, self.goal_size, self.goal_size], rgbaColor=[1, 0, 0, 0.5])
        self.goal = p.createMultiBody(baseVisualShapeIndex=self.goal_visual, basePosition=self.goal_pos, physicsClientId=self.client)
        agent_pos = list(p.getBasePositionAndOrientation(self.agent, physicsClientId=self.client)[0])
        wall_distances = self.get_wall_distances(agent_pos)  # 벽과의 거리 계산
        self.state = agent_pos + self.goal_pos + wall_distances
        self.previous_position = agent_pos  # 이전 위치 저장
        return np.array(self.state)

    def get_wall_distances(self, agent_pos):
        # 벽과의 거리를 계산하는 함수
        distances = []
        for wall in self.walls:
            start, end = np.array(wall[0]), np.array(wall[1])
            distance = np.linalg.norm(np.cross(end - start, start - np.array(agent_pos)) / np.linalg.norm(end - start))
            distances.append(distance)
        return distances

    def create_wall(self, start_pos, end_pos):
        length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        orientation = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        mid_pos = [(start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2, (start_pos[2] + end_pos[2]) / 2]
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[length / 2, 0.05, self.wall_height / 2])
        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, 0.05, self.wall_height / 2])
        p.createMultiBody(baseVisualShapeIndex=wall_visual, baseCollisionShapeIndex=wall_collision, basePosition=mid_pos,
                          baseOrientation=p.getQuaternionFromEuler([0, 0, orientation]), physicsClientId=self.client)

    def create_borders(self):
        borders = [
            [(0, 0, 0), (3, 0, 0)],  # 바닥 경계
            [(0, 3, 0), (3, 3, 0)],
            [(0, 0, 0), (0, 3, 0)],
            [(3, 0, 0), (3, 3, 0)],
            [(0, 0, 3), (3, 0, 3)],  # 천장 경계
            [(0, 3, 3), (3, 3, 3)],
            [(0, 0, 3), (0, 3, 3)],
            [(3, 0, 3), (3, 3, 3)],
            [(0, 0, 0), (0, 0, 3)],  # 벽 경계
            [(3, 0, 0), (3, 0, 3)],
            [(0, 3, 0), (0, 3, 3)],
            [(3, 3, 0), (3, 3, 3)],
        ]
        for border in borders:
            self.create_wall(border[0], border[1])

    def step(self, action):
        force = 20000  # 힘 적용, 힘을 증가시켜 이동이 잘 되도록 함
        if action == 0:  # forward
            p.applyExternalForce(self.agent, -1, [force, 0, 0], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client)
        elif action == 1:  # backward
            p.applyExternalForce(self.agent, -1, [-force, 0, 0], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client)
        elif action == 2:  # left
            p.applyExternalForce(self.agent, -1, [0, force, 0], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client)
        elif action == 3:  # right
            p.applyExternalForce(self.agent, -1, [0, -force, 0], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client)
        elif action == 4:  # up
            p.applyExternalForce(self.agent, -1, [0, 0, force], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client)
        elif action == 5:  # down
            p.applyExternalForce(self.agent, -1, [0, 0, -force], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client)

        for _ in range(10):  # 더 많은 시뮬레이션 단계 실행
            p.stepSimulation(physicsClientId=self.client)

        agent_pos = list(p.getBasePositionAndOrientation(self.agent, physicsClientId=self.client)[0])
        wall_distances = self.get_wall_distances(agent_pos)  # 벽과의 거리 계산
        next_state = agent_pos + self.goal_pos + wall_distances
        distance = np.linalg.norm(np.array(agent_pos[:3]) - np.array(self.goal_pos[:3]))

        contact_points = p.getContactPoints(self.agent, physicsClientId=self.client)
        reward = -distance
        done = False

        if contact_points:  # 벽과 충돌했는지 확인
            reward -= 100  # 큰 페널티 부여

        if all(abs(agent_pos[i] - self.goal_pos[i]) <= self.goal_size for i in range(3)):  # 목표 영역에 도달했는지 확인
            reward = 100
            done = True

        # 추가 보상: 에이전트가 목표 지점에 더 가까워지면 추가 보상 제공
        if distance < 1.0:
            reward += 50 * (1.0 - distance)

        return np.array(next_state), reward, done, {}

    def render(self, mode='rgb_array'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[1.5, 1.5, 1.5],
                                                          distance=5,
                                                          yaw=50,
                                                          pitch=-35,
                                                          roll=0,
                                                          upAxisIndex=2,
                                                          physicsClientId=self.client)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=1.0,
                                                   nearVal=0.1,
                                                   farVal=100.0,
                                                   physicsClientId=self.client)
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=640,
                                                                   height=480,
                                                                   viewMatrix=view_matrix,
                                                                   projectionMatrix=proj_matrix,
                                                                   physicsClientId=self.client)
        return rgbImg

    def close(self):
        p.disconnect(self.client)

# 폴더 생성
if not os.path.exists('episodeImages'):
    os.makedirs('episodeImages')

# PPO 학습 및 실행
env = EndoscopeEnv()
state_shape = env.observation_space.shape
action_space = env.action_space.n

actor = build_actor(state_shape, action_space)
critic = build_critic(state_shape)

# 모델 컴파일 추가
actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate_actor))
critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate_critic), loss='mse')

actor_optimizer = tf.keras.optimizers.Adam(learning_rate_actor)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate_critic)

memory = Memory()

# 이동 경로 시각화를 위한 데이터 저장
episode_paths = []
max_steps = 50  # 스텝 수 제한

def plot_path_3d(path, episode):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o')
    ax.scatter(0.5, 0.5, 0.5, color='black', s=100)  # 시작점 표시
    ax.scatter(2.5, 2.5, 2.5, color='red', s=100)  # 도착점 표시
    ax.set_title(f'Episode {episode} Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f'episodeImages/episode_{episode}_path.png')
    plt.close()

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    path = [state[:3]]
    step_count = 0

    while not done and step_count < max_steps:  # 스텝 수 제한 추가
        action, log_prob = choose_action(state, actor)
        next_state, reward, done, _ = env.step(action)
        memory.store(state, action, reward, next_state, done, log_prob)
        state = next_state
        score += reward
        path.append(state[:3])
        step_count += 1  # 스텝 수 증가

    last_position = path[-1]
    print(f"Episode: {episode}, Score: {score}, Steps: {step_count}, Last Position: {last_position}")  # 스텝 수 및 마지막 위치 출력
    train(memory, actor, critic, actor_optimizer, critic_optimizer, batch_size, epochs, gamma, lambda_gae, epsilon)
    episode_paths.append(path)
    plot_path_3d(path, episode)  # 각 에피소드가 끝날 때마다 경로 시각화

env.close()