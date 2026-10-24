"""
DevFest demo: an AI that learns a game from nothing but pixels and a score.

The game is "Catch": a ball falls from the top of a tiny 10x5 pixel screen,
and the agent moves a paddle along the bottom to catch it.
The agent is never told the rules. It sees only the screen (50 pixels)
and the score (+1 for a catch, -1 for a miss).

This is a from-scratch Deep Q-Network (DQN), the same recipe DeepMind used
for Atari in 2015, shrunk down to run on a laptop in under a minute with only NumPy:
  - a neural network that looks at pixels and scores each possible action
  - experience replay: learn from a memory of past moments, shuffled
  - a target network: a slowly updated copy, so the agent isn't chasing itself

Run:  python dqn_catch.py
      python dqn_catch.py --watch           (also print games before and after training)
      python dqn_catch.py --episodes 300    (shorter run, for a tight demo slot)
      python dqn_catch.py --seed 1          (a different run of the same experiment)
      python dqn_catch.py --test            (check the rules and the learning still hold)
"""
import argparse
import numpy as np

ROWS, COLS = 10, 5
ACTIONS = ["left", "stay", "right"]
rng = np.random.default_rng(0)


def seed(value):
    """Reseed everything: the game, the network's first weights, and the agent's dice."""
    global rng
    rng = np.random.default_rng(value)


# ------------------------------------------------------------------ the game
class Catch:
    def reset(self):
        self.ball_r, self.ball_c = 0, int(rng.integers(COLS))
        self.paddle = int(rng.integers(COLS))
        return self.screen()

    def screen(self):
        s = np.zeros((ROWS, COLS), dtype=np.float32)
        s[self.ball_r, self.ball_c] = 1.0
        s[ROWS - 1, self.paddle] += 0.5               # paddle pixels are dimmer than the ball
        return s

    def step(self, action):
        self.paddle = int(np.clip(self.paddle + action - 1, 0, COLS - 1))
        self.ball_r += 1
        if self.ball_r == ROWS - 1:                       # ball reached the bottom
            reward = 1.0 if self.ball_c == self.paddle else -1.0
            return self.screen(), reward, True
        return self.screen(), 0.0, False


def draw(s):
    glyph = {0.0: "·", 1.0: "●", 0.5: "▬", 1.5: "◉"}   # ◉ = ball landed on the paddle
    return "\n".join("".join(glyph[float(v)] for v in row) for row in s)


# ------------------------------------------------------------------ the brain
class QNetwork:
    """A tiny neural network: 50 pixels in -> 64 hidden units -> one score per action."""
    def __init__(self, n_in=ROWS * COLS, n_hidden=64, n_out=3):
        self.p = {
            "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)), "b2": np.zeros(n_out),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}   # Adam optimiser state
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        h = np.maximum(0, x @ self.p["W1"] + self.p["b1"])
        return h @ self.p["W2"] + self.p["b2"], h

    def q(self, x):
        return self.forward(x)[0]

    def train_step(self, x, actions, targets, lr=1e-3):
        """Move Q(x, action) toward the target, for the actions actually taken."""
        q, h = self.forward(x)
        err = np.zeros_like(q)
        idx = np.arange(len(x))
        err[idx, actions] = q[idx, actions] - targets
        err = np.clip(err, -1, 1) / len(x)            # Huber-style clipping, as in DQN
        grads = {"W2": h.T @ err, "b2": err.sum(0)}
        dh = (err @ self.p["W2"].T) * (h > 0)
        grads.update({"W1": x.T @ dh, "b1": dh.sum(0)})
        self.t += 1
        for k in self.p:                               # Adam update
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            m_hat = self.m[k] / (1 - 0.9 ** self.t)
            v_hat = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def copy_from(self, other):
        self.p = {k: v.copy() for k, v in other.p.items()}


# ------------------------------------------------------------------ memory
class ReplayMemory:
    """Remembers past moments. Learning from random old moments breaks the
    'everything I just saw is related' problem that makes naive training unstable."""
    def __init__(self, size=5000):
        self.s = np.zeros((size, ROWS * COLS), np.float32); self.s2 = np.zeros_like(self.s)
        self.a = np.zeros(size, int); self.r = np.zeros(size); self.d = np.zeros(size)
        self.size, self.n, self.i = size, 0, 0

    def add(self, s, a, r, s2, done):
        self.s[self.i], self.a[self.i], self.r[self.i] = s.ravel(), a, r
        self.s2[self.i], self.d[self.i] = s2.ravel(), float(done)
        self.i = (self.i + 1) % self.size; self.n = min(self.n + 1, self.size)

    def sample(self, k):
        j = rng.integers(self.n, size=k)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


# ------------------------------------------------------------------ play & learn
def play(net, games=200, epsilon=0.0):
    env, score = Catch(), 0.0
    for _ in range(games):
        s, done = env.reset(), False
        while not done:
            a = int(rng.integers(3)) if rng.random() < epsilon else int(np.argmax(net.q(s.ravel()[None])))
            s, r, done = env.step(a)
        score += r
    return (score / games + 1) / 2                     # fraction of balls caught


def show_game(net, title):
    env = Catch(); s, done = env.reset(), False
    print(f"\n--- {title} ---")
    frames = [s]
    while not done:
        s, r, done = env.step(int(np.argmax(net.q(s.ravel()[None]))))
        frames.append(s)
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        print(draw(f)); print()
    print("CAUGHT IT!" if r > 0 else "Missed.")


def train(episodes=800, watch=False, quiet=False):
    env, net, target, memory = Catch(), QNetwork(), QNetwork(), ReplayMemory()
    target.copy_from(net)
    gamma, batch, steps = 0.99, 32, 0
    checkpoints = {ep for ep in (50, 100, 200, 300, 500) if ep < episodes} | {episodes}
    if not quiet:
        print(f"Random button-mashing catches {play(net, epsilon=1.0):.0%} of balls")
        print(f"Before training:  catches {play(net):.0%} of balls")
    if watch: show_game(net, "Before training")

    for ep in range(1, episodes + 1):
        s, done = env.reset(), False
        while not done:
            epsilon = max(0.05, 1.0 - steps / 4000)                 # explore a lot early, less later
            a = int(rng.integers(3)) if rng.random() < epsilon else int(np.argmax(net.q(s.ravel()[None])))
            s2, r, done = env.step(a)                                # ACT
            memory.add(s, a, r, s2, done)                            # REMEMBER
            s, steps = s2, steps + 1
            if memory.n >= 200:                                      # LEARN from random memories
                bs, ba, br, bs2, bd = memory.sample(batch)
                targets = br + gamma * (1 - bd) * target.q(bs2).max(axis=1)   # Bellman target
                net.train_step(bs, ba, targets)
            if steps % 200 == 0:
                target.copy_from(net)                                # UPDATE the slow copy
        if ep in checkpoints and not quiet:
            print(f"After {ep:>4} games: catches {play(net):.0%} of balls   (exploration {epsilon:.0%})")

    if watch: show_game(net, "After training")
    return net


# ------------------------------------------------------------------ self-check
def self_test():
    """The rules the agent is never told, plus proof that it still learns them."""
    seed(0)
    env = Catch()
    s = env.reset()
    assert s.shape == (ROWS, COLS), f"the screen is {ROWS}x{COLS} pixels, got {s.shape}"
    assert s.sum() == 1.5, "one ball pixel (1.0) and one paddle pixel (0.5)"

    env.reset(); env.ball_c, env.paddle = COLS - 1, 0      # chase the ball all the way right
    done = False
    while not done: _, r, done = env.step(2)
    assert r == 1.0, f"paddle under the ball must score +1, got {r}"

    env.reset(); env.ball_c, env.paddle = 0, 0             # run away from the ball instead
    done = False
    while not done: _, r, done = env.step(2)
    assert r == -1.0, f"paddle away from the ball must score -1, got {r}"

    seed(0)
    rate = play(train(episodes=300, quiet=True), games=200)
    assert rate > 0.85, f"300 games should beat random button-mashing by a mile, got {rate:.0%}"
    print(f"All checks passed. After 300 games the agent catches {rate:.0%} of balls.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Watch a Deep Q-Network learn Catch from pixels and a score.")
    ap.add_argument("--watch", action="store_true", help="print a game before and after training")
    ap.add_argument("--episodes", type=int, default=800, help="games to train on (default: 800)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    ap.add_argument("--test", action="store_true", help="run the self-check and exit")
    args = ap.parse_args()
    if args.test:
        self_test()
    else:
        seed(args.seed)
        train(episodes=args.episodes, watch=args.watch)
