[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/qZAaOmzv)
# HW2 - Tabular Methods

## What to do

Your goal is to implement the following model-free RL algorithms in a "tabular" setting (i.e., assuming small finite state and action spaces):
- On-policy first-visit MC control (Chapter 5.4, Sutton and Barto)
- SARSA (on-policy TD(0), Chapter 6.4, Sutton and Barto)
- Q-learning (off-policy TD(0), Chapter 6.5, Sutton and Barto)

You will test your algorithms in the Frozen Lake MDP described here: https://gymnasium.farama.org/environments/toy_text/frozen_lake/. Since you are using model-free methods, you WILL use the Frozen Lake gymnasium environment this time. Assume the following for the MDP.
- Full observability, an infinite time horizon, and a discount factor of $\gamma = 0.95$
- A map defined by "4x4":["SFFF", "FHFH", "FFFH", "HFFG"]
- `is_slippery=True` and `is_slippery=False`

More specifically, please do the following:
1. Fully define your MDP (i.e., the state space, action space, etc.)
1. Plot the evaluation return versus the number of time steps. See Chapter 2.7, Albrecht et al. "MARL" for more details on plotting evaluation returns.
1. Plot the policy or value function that corresponds to each trained agent.

## What to submit

### 1. A report and code

Commit your code and a report (titled `hwX-netid.pdf`) to your repository. Then submit your repository to Gradescope.

The report should be no longer than 6 pages (NOT including references) and formatted using either IEEE or AIAA conference paper format. The report should include the following sections (and content), at a minimum:
- introduction (e.g., problem and model)
- methods (e.g., algorithm pseudocode, any specific details needed to understand your algorithm implementations like hyperparameters)
- results
- conclusions
- references

### 2. Pull request

Create a [pull request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) to enable a review of your code by a colleague in the class.

You can do this by:
1. View your repository in a web browser.
2. Click the "Edit file" icon for the `README.md` file.
3. Make a small edit to the file. For example, add a line at the bottom that says "Edit for final PR."
4. Click the "Commit changes..." button.
5. MAKE SURE TO SELECT THE "Create a new branch for this commit and start a pull request" option.
6. Click the "Propose changes" button.
7. Click the "Create pull request" button.
8. Click the "Create pull request" button again on the next page.

## Resources

Here are some resources that may be helpful:
- A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)
