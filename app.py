"""Streamlit dashboard for the DQN in dqn_catch.py.

The agent, the game and the maths all live in dqn_catch. This file only
draws them: the screen the agent sees, the value it puts on each move,
and the catch rate climbing as it learns.

Run:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Imported as a module, not by value: dqn_catch.seed() rebinds dqn_catch.rng,
# and `from dqn_catch import rng` would keep pointing at the old generator.
import dqn_catch
from dqn_catch import Catch, QNetwork, ReplayMemory, play, ROWS, COLS, ACTIONS

st.set_page_config(
    page_title="Catch | Reinforcement Learning Lab",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Catch: An Agent That Learns")
st.caption(
    "Deep Q-Learning from pixels and rewards. "
    "No rules given to the agent."
)

# --------------------------- Session state


def new_agent(seed=None):
    """Start over: fresh network, empty memory, nothing learned."""
    if seed is not None:
        dqn_catch.seed(seed)

    st.session_state.net = QNetwork()
    st.session_state.target = QNetwork()
    st.session_state.target.copy_from(st.session_state.net)
    st.session_state.memory = ReplayMemory()
    st.session_state.steps = 0
    st.session_state.episode = 0
    st.session_state.catch_history = []
    st.session_state.epsilon_history = []
    st.session_state.epsilon = 1.0


if "net" not in st.session_state:
    new_agent()

net = st.session_state.net
target = st.session_state.target
memory = st.session_state.memory

# --------------------------- Controls

with st.sidebar:
    st.header("Experiment controls")

    episodes_to_train = st.slider(
        "Episodes per run",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    seed_value = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=0,
        help="Applied when you reset the agent.",
    )

    show_live = st.checkbox(
        "Show live gameplay during training",
        value=True,
    )

    st.caption(
        "Live rendering is slower. Turn it off "
        "to train faster."
    )

    if st.button("Reset agent", width="stretch"):
        new_agent(seed=int(seed_value))
        st.rerun()

# --------------------------- Game rendering

def render_game(screen, title="Catch"):
    fig, ax = plt.subplots(figsize=(4, 5))

    ax.imshow(
        screen,
        cmap="viridis",
        vmin=0,
        vmax=1.5,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(-0.5, COLS, 1))
    ax.set_yticks(np.arange(-0.5, ROWS, 1))
    ax.grid(color="white", linewidth=1, alpha=0.35)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, color="white", pad=12)

    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")
    ax.tick_params(length=0)

    fig.tight_layout()
    return fig


def action_name(action):
    return ACTIONS[action].upper()


def q_table(q_values):
    """Q-values keyed by move name, so the bar chart reads LEFT / STAY / RIGHT."""
    return {
        "Q-value": {
            action_name(i): float(v)
            for i, v in enumerate(q_values)
        }
    }


# --------------------------- Dashboard

left, right = st.columns([1, 1])

with left:
    st.subheader("Live environment")
    game_slot = st.empty()
    status_slot = st.empty()

with right:
    st.subheader("Agent's decision")
    q_slot = st.empty()
    action_slot = st.empty()
    metrics_slot = st.empty()

st.divider()

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Catch rate over training")
    catch_chart = st.empty()

with chart_right:
    st.subheader("Exploration over training")
    epsilon_chart = st.empty()

st.divider()
st.subheader("Training progress")
progress_slot = st.empty()
log_slot = st.empty()

# Display the current model before training.
env = Catch()
screen = env.reset()

q_values = net.q(screen.ravel()[None, :])[0]
best_action = int(np.argmax(q_values))

with game_slot:
    fig = render_game(screen, "Current environment")
    st.pyplot(fig)
    plt.close(fig)

with q_slot:
    st.bar_chart(q_table(q_values))

with action_slot:
    st.info(
        f"Current preferred action: "
        f"{action_name(best_action)}"
    )

def draw_metrics():
    # st.empty() holds one element, so the two metrics go in a container inside it.
    with metrics_slot.container():
        st.metric("Episodes completed", st.session_state.episode)
        st.metric(
            "Exploration rate",
            f"{st.session_state.epsilon:.0%}",
        )


draw_metrics()

if st.session_state.catch_history:
    with catch_chart:
        st.line_chart(
            {"Catch rate": st.session_state.catch_history}
        )

if st.session_state.epsilon_history:
    with epsilon_chart:
        st.line_chart(
            {"Epsilon": st.session_state.epsilon_history}
        )

# --------------------------- Train

if st.button(
    "▶ Train agent",
    type="primary",
    width="stretch",
):
    gamma = 0.99
    batch_size = 32
    start_episode = st.session_state.episode
    total_episodes = episodes_to_train

    progress = progress_slot.progress(0.0)
    log = []

    for local_ep in range(1, total_episodes + 1):
        screen = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            steps = st.session_state.steps

            epsilon = max(
                0.05,
                1.0 - steps / 4000,
            )

            if dqn_catch.rng.random() < epsilon:
                action = int(dqn_catch.rng.integers(3))
            else:
                action = int(
                    np.argmax(
                        net.q(screen.ravel()[None, :])[0]
                    )
                )

            next_screen, reward, done = env.step(action)

            memory.add(
                screen,
                action,
                reward,
                next_screen,
                done,
            )

            episode_reward += reward
            screen = next_screen

            st.session_state.steps += 1
            st.session_state.epsilon = epsilon

            # Learn from a random batch of past experiences.
            if memory.n >= 200:
                bs, ba, br, bs2, bd = memory.sample(
                    batch_size
                )

                targets = (
                    br
                    + gamma
                    * (1 - bd)
                    * target.q(bs2).max(axis=1)
                )

                net.train_step(bs, ba, targets)

            # Update the target network periodically.
            if st.session_state.steps % 200 == 0:
                target.copy_from(net)

            # Render actual gameplay frames.
            if show_live:
                fig = render_game(
                    screen,
                    f"Episode {start_episode + local_ep}",
                )

                game_slot.pyplot(fig)
                plt.close(fig)

                current_q = net.q(
                    screen.ravel()[None, :]
                )[0]

                q_slot.bar_chart(q_table(current_q))

                action_slot.info(
                    f"Latest action: {action_name(action)}"
                )

                status_slot.caption(
                    f"Reward: {reward:+.0f} | "
                    f"Step: {st.session_state.steps}"
                )

        st.session_state.episode += 1
        st.session_state.epsilon_history.append(epsilon)

        # Evaluate every 10 episodes and at the end.
        should_evaluate = (
            local_ep % 10 == 0
            or local_ep == total_episodes
        )

        if should_evaluate:
            # play() with epsilon=0: no exploring, no learning, pure skill.
            catch_rate = play(net, games=30)

            st.session_state.catch_history.append(
                catch_rate
            )

            log.append(
                {
                    "Episode": st.session_state.episode,
                    "Catch rate": f"{catch_rate:.0%}",
                    "Reward": f"{episode_reward:+.0f}",
                    "Epsilon": f"{epsilon:.2f}",
                }
            )

            catch_chart.line_chart(
                {"Catch rate": st.session_state.catch_history}
            )

            epsilon_chart.line_chart(
                {"Epsilon": st.session_state.epsilon_history}
            )

            log_slot.dataframe(
                log,
                width="stretch",
                hide_index=True,
            )

        progress.progress(local_ep / total_episodes)

    progress_slot.progress(1.0)
    draw_metrics()        # the counters above were drawn before this run started
    st.success("Training batch completed!")

    final_rate = play(net, games=100)

    st.metric(
        "Final evaluation catch rate",
        f"{final_rate:.0%}",
        help="Measured on 100 evaluation games without exploration.",
    )

    st.balloons()
