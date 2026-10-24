# Nothing But Pixels and a Score

How an AI taught itself to play Atari, and what it means for today's agents.
A talk by Brian Kayongo for DevFest Kigali 2026.

An agent is dropped into a game called **Catch**. A ball falls down a tiny
10×5 screen; a paddle sits at the bottom. The agent is never told that a ball
exists, that a paddle exists, or what winning means. It sees 50 numbers and
receives `+1` when it catches and `−1` when it misses. From that alone it
learns to catch every ball.

---

## What's here

| File | What it is |
| --- | --- |
| `talk-script.md` | Proposal text and the full 30-minute speaker script |
| `watch-the-ai-learn.html` | The visual demo: watch the AI play Catch, get scored and learn |
| `dqn_catch.py` | The same AI in Python, printing the game as text in the terminal |
| `app.py` | Optional Streamlit dashboard: the same agent with charts and sliders |
| `requirements.txt` | Dependencies for the two Python programs |

The HTML and the Python are **independent implementations of the same
algorithm**. Neither imports the other. Both reach a 100% catch rate. If one
fails on the day, the other is a complete replacement.

---

## Quick start

Three ways in, cheapest first.

```bash
# 1. The browser demo. No install, nothing to run.
open watch-the-ai-learn.html
```

```bash
# 2. The Python demo. Needs the setup in "Python setup" below first.
python dqn_catch.py --watch
```

```bash
# 3. The dashboard. Same setup, then:
streamlit run app.py
```

---

## The browser demo

```bash
open watch-the-ai-learn.html
```

`open` is the macOS command that hands a file to its default application; on
Linux use `xdg-open`, on Windows `start`. You can equally drag the file onto a
browser window. There is no server, no build step and no install — the whole
demo, neural network included, is one self-contained HTML file.

### What you are looking at

| Panel | What it shows |
| --- | --- |
| **The arcade screen** | The game. Yellow square is the ball, blue bar is the paddle. |
| **What the AI sees** | The literal 50 numbers fed to the network. `1` is the ball, `.5` the paddle, `1.5` both in one cell. |
| **What the AI is thinking** | The network's score for each of the three moves. The highlighted bar is the one it would pick. |
| **Scoreboard** | Games played, caught, missed, all-time. |
| **Catch rate, last 50** | Rolling skill. This is the number that climbs. |
| **Random moves** | How often it is still ignoring what it knows and pressing a random button. Starts at 100%, decays to 5%. |
| **Memories stored** | Past moments held for replay. Caps at 5,000. |
| **Brain updates** | How many times the network's weights have been nudged. |
| **Learning curve** | Catch rate over games. The dashed line is random button-mashing, about 20%. |

### Driving it on stage

1. Press **Play** on **Watch** speed. A few clumsy, random games. Point at
   "Random moves: 100%" — it is not thinking yet, it is flailing.
2. Switch to **Turbo**. The curve climbs. This takes a few seconds, not minutes.
3. Switch back to **Watch**, press **Test the AI**. Exploration and learning
   both switch off, so what you see is purely what it has learned. It catches
   every ball.
4. Press **New untrained AI** to wipe it and start over.

| Key | Does |
| --- | --- |
| `Space` | Play or pause |
| `1` `2` `3` | Watch / Fast / Turbo speed |
| `T` | Test the AI (no random moves, no learning) |
| `R` | New untrained AI |

**Offline:** the page's single network request is a Google Fonts stylesheet for
the arcade headings. It loads with `display=swap` and falls back to a monospace
font, so everything works with the wifi off — which is how to assume it will
run.

---

## Python setup

Do this once, before the talk. Each line explained.

```bash
cd "path/to/nothing-but-pixels"
```
Work from the project folder, so `python dqn_catch.py` finds the file.

```bash
python3 -m venv .venv
```
Create a virtual environment: a private Python install in a `.venv` folder, so
these packages never collide with anything else on the machine. Run it once;
after that the folder exists.

```bash
source .venv/bin/activate
```
Activate it. Your prompt gains a `(.venv)` prefix, and from here `python` and
`pip` mean the ones inside `.venv`. **This is per terminal window** — open a new
tab and you activate again. On Windows the line is `.venv\Scripts\activate`.

> This machine has no bare `python` on the PATH, only `python3`. Inside an
> activated venv, `python` exists and is the right one. That is the main reason
> to bother with the venv rather than installing globally.

```bash
pip install -r requirements.txt
```
Install the dependencies listed in `requirements.txt`:

| Package | Needed by | Why |
| --- | --- | --- |
| `numpy` | `dqn_catch.py` | The only dependency of the talk demo. All the matrix maths. |
| `streamlit` | `app.py` | The dashboard's web UI. Pinned `>=1.66` for the `width="stretch"` argument that replaced `use_container_width`, which Streamlit removed after 2025-12-31. |
| `matplotlib` | `app.py` | Draws the game screen as a heatmap in the dashboard. |

If you only want the terminal demo, `pip install numpy` is enough.

```bash
python dqn_catch.py --test
```
Confirm it all works. Takes about a second. Expect:

```
All checks passed. After 300 games the agent catches 100% of balls.
```

---

## Running the Python demo

```bash
python dqn_catch.py
```

Trains for 800 games in **three or four seconds** and prints its progress.

### Reading the output, line by line

```
Random button-mashing catches 22% of balls
```
The floor. Press buttons at random and you catch roughly one ball in five, by
luck. Every later number is measured against this.

```
Before training:  catches 20% of balls
```
The untrained network, playing its best. Same as random — because its weights
are random. It has learned nothing yet.

```
After   50 games: catches 6% of balls   (exploration 89%)
```
**Worse than random, and this is not a bug.** At 89% exploration almost every
move is still a coin flip, and the little it has learned so far is a wrong first
guess that it is briefly committed to. Learning is not monotonic. Worth saying
out loud on stage, because the audience will see the dip.

```
After  100 games: catches 33% of balls   (exploration 78%)
After  200 games: catches 82% of balls   (exploration 55%)
After  300 games: catches 100% of balls   (exploration 33%)
After  500 games: catches 100% of balls   (exploration 5%)
After  800 games: catches 100% of balls   (exploration 5%)
```
It climbs past random, then saturates. Note the two columns move together:
as the catch rate rises, exploration falls — it stops guessing because guessing
has become worse than using what it knows.

The catch-rate figures are measured with exploration switched **off**, so they
report skill, not luck.

### Flags

| Flag | What it does |
| --- | --- |
| `--watch` | Also print a game before and after training, drawn as text |
| `--episodes N` | Train on N games instead of 800 |
| `--seed N` | Same experiment, different dice |
| `--test` | Run the self-check and exit |
| `--help` | List all of the above |

```bash
python dqn_catch.py --watch
```
Prints three frames of one game before training and three after. A `●` is the
ball, `▬` the paddle, `◉` both in one cell — which is exactly what a catch
looks like. The run ends with `CAUGHT IT!` or `Missed.`

```bash
python dqn_catch.py --episodes 300
```
Shorter run for a tight slot. 300 games is enough to reach 100%; the remaining
500 only hold it there.

```bash
python dqn_catch.py --seed 7
```
Everything random in the program — the ball's column, the network's starting
weights, the exploration dice — comes from one seeded generator. Same seed,
same run, every time. **If a live run goes badly in front of the room, change
this number.** Seeds `0` and `7` both reach 100% by game 300.

```bash
python dqn_catch.py --test
```
The self-check. It asserts the screen is one ball plus one paddle, that a paddle
under the ball scores `+1` and away from it scores `−1`, then trains a fresh
agent for 300 games and fails if it has not comfortably beaten random
button-mashing. About one second. Run it before you walk on.

---

## Running the dashboard

```bash
streamlit run app.py
```

Starts a local web server and opens the dashboard at
<http://localhost:8501>. Stop it with `Ctrl+C`.

The same agent as `dqn_catch.py` — `app.py` imports it rather than
reimplementing it — with a live view of the screen, the value it puts on each
move, and the catch rate climbing.

| Sidebar control | What it does |
| --- | --- |
| **Episodes per run** | How many games one press of **Train agent** plays, 10 to 200 |
| **Random seed** | Applied when you press **Reset agent** |
| **Show live gameplay** | Draw every frame while training |
| **Reset agent** | Throw away everything learned and start over |

Press **▶ Train agent** to run a batch. Press it again to train further — the
agent keeps what it learned between batches.

**On speed:** with live gameplay on, the dashboard draws a full matplotlib
figure for every single step. That is what makes it slow, and it is the point —
you can watch it play. Turn it off to train in a fraction of the time.

Use this one to explore. For the talk itself, `dqn_catch.py` is faster to read
on a projector and has nothing to go wrong.

---

## Before the talk

```bash
cd "path/to/nothing-but-pixels"
source .venv/bin/activate
python dqn_catch.py --test
```

- [ ] `--test` prints **All checks passed**
- [ ] `watch-the-ai-learn.html` opens and **Play** runs — with the wifi off
- [ ] **Test the AI** catches every ball
- [ ] Terminal font is large enough to read from the back
- [ ] You know which seed you are using

---

## How it works

A Deep Q-Network (DQN), the method DeepMind used to learn Atari games in 2015,
built from scratch with no deep learning library:

- a small neural network that looks at the screen and scores each possible move
- a memory of past moments, sampled at random to learn from
- a slowly updated copy of the network, so training stays stable

The AI is never told the rules. It sees 50 numbers (the screen) and gets +1 for
a catch and −1 for a miss.

### The loop

Every step of every game does four things. They are commented `ACT`, `REMEMBER`,
`LEARN` and `UPDATE` in both implementations, so you can point at them:

1. **ACT** — look at the screen, score all three moves, play the best one. Early
   on, mostly ignore that and press something random instead.
2. **REMEMBER** — file the moment away: what the screen was, what was pressed,
   what came back, what the screen became.
3. **LEARN** — pull 32 random moments out of that memory and nudge the network
   toward the reward each one actually earned. Random, because learning from
   moments in the order they happened is unstable: they are all too alike.
4. **UPDATE** — every 200 steps, copy the network into a frozen twin. Scores are
   measured against the twin, so the network is not chasing its own moving
   target.

### The map

| Part | `dqn_catch.py` | `watch-the-ai-learn.html` |
| --- | --- | --- |
| The game | `class Catch` | `class Catch` |
| The network | `class QNetwork` | `class QNet` |
| The memory | `class ReplayMemory` | `Agent.memory` |
| The loop | `train()` | `Agent.step()` |
| Playing without learning | `play()` | `Agent.testMode` |

The network is 50 inputs → 64 hidden units → 3 scores, one per move. ReLU
activation, Adam optimiser, both written out by hand. Rewards are discounted at
0.99, errors clipped to ±1, memory holds 5,000 moments, exploration decays from
100% to 5% over 4,000 steps.
# Nothing-But-Pixels-and-a-Score
