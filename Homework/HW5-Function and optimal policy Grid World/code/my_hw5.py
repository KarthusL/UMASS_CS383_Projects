from gridworld import GridWorld, GridWorld_MDP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def policy_iteration(mdp, gamma=1, iters=5, plot=True):
    '''
    Performs policy iteration on an mdp and returns the value function and policy 
    :param mdp: mdp class (GridWorld_MDP) 
    :param gam: discount parameter should be in (0, 1] 
    :param iters: number of iterations to run policy iteration for
    :param plot: boolean for if a plot should be generated for the utilities of the start state
    :return: two numpy arrays of length |S| and one of length iters.
    Two arrays of length |S| are U and pi where U is the value function
    and pi is the policy. The third array contains U(start) for each iteration
    the algorithm.
    '''

    pi = np.zeros(mdp.num_states, dtype=np.int)     #Policy
    U = np.zeros(mdp.num_states)  #V
    Ustart = []
    #TODO Implement policy iteration
    S, A, R, P, P_next = mdp.S, mdp.A, mdp.R, mdp.P, mdp.P_snexts
    I = np.eye(mdp.num_states, mdp.num_states)
    b = [R(s) for s in S()]
    prob = np.zeros((mdp.num_states, mdp.num_states))

    while iters != 0:
        iters -= 1
        prob = initial_matrix(mdp, pi)                  #algorithm on PPT
        A = I - gamma * prob
        pse = np.linalg.pinv(A)
        U = np.dot(pse, b)
        for state in mdp.S():
            best_action = pi[state]
            best_value = sum([mdp.P(s, state, pi[state]) * U[s] for s in mdp.S()])
            for action in mdp.A(state):
                temp_value = sum([mdp.P(s, state, action) * U[s] for s in mdp.S()])
                if temp_value > best_value:
                    best_action = action
                    best_value = temp_value
            pi[state] = best_action
        startin = mdp.loc2state[mdp.start]
        Ustart.append(U[startin])

        # for state1 in range(mdp.num_states):
        #     for action in A(state0):
        #         print(state1, state0, P_next(state0, action))
        #         prob[state0][state1] =  P_next(state0, action)


    # while iters != 0:
    #     iters -= 1
    #     for state in range(mdp.num_states):
    #         print(state)
    #         print(type(P_next(state, key).values()) for key in A(state))
    #         U[state] = R(state) + (gamma * (sum(P_next(state, key).values() * U[state]) for key in A(state)))
    #
    #     for state in range(mdp.num_states):
    #         temp = sum(P_next(state, key).values() )
    #
    # print(U)
    if plot:
        fig = plt.figure()
        plt.title("Policy Iteration with $\gamma={0}$".format(gamma))
        plt.xlabel("Iteration (k)")
        plt.ylabel("Utility of Start")
        plt.ylim(-1, 1)
        plt.plot(Ustart)
        pp = PdfPages('./plots/piplot.pdf')
        fig.show()
        pp.savefig(fig)
        plt.close()
        pp.close()

    print("U\n", U)
    print("pi\n", pi)
    print("Ustart\n", Ustart)
    #U and pi should be returned with the shapes and types specified
    return U, pi, np.array(Ustart)

def initial_matrix(mdp, policy):
    matrix = np.zeros((mdp.num_states, mdp.num_states))
    for state in mdp.S():
        renew_data(mdp, matrix, state, policy[state])
    return matrix

def renew_data(mdp, matrix, state, policy):
    P_next = mdp.P_snexts
    dictionary = P_next(state, policy).items()
    for key, value in dictionary:
        if not mdp.is_absorbing(key):
            matrix[state][key] = value
    return matrix



def td_update(v, s1, r, s2, terminal, alpha, gamma):
    '''
    Performs the TD update on the value function v for one transition (s,a,r,s').
    Update to v should be in place.
    :param v: The value function, a numpy array of length |S|
    :param s1: the current state, an integer 
    :param r: reward for the transition
    :param s2: the next state, an integer
    :param terminal: bool for if the episode ended
    :param alpha: learning rate parameter
    :param gamma: discount factor
    :return: Nothing
    '''
    #TODO implement the TD Update
    #you should update the value function v inplace (does not need to be returned)
    if not terminal:
        v[s1] = v[s1] + alpha * (r + gamma * v[s2] - v[s1])
    else:
        v[s1] = v[s1] + alpha * (r - v[s1])
    pass


def td_episode(env, pi, v, gamma, alpha, max_steps=1000):
    '''
    Agent interacts with the environment for one episode update the value function after
    each iteration. The value function update should be done with the TD learning rule.
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S| representing the policy
    :param v: numpy array of length |S| representing the value function
    :param gamma: discount factor
    :param alpha: learning rate
    :param max_steps: maximum number of steps in the episode
    :return: two floats G, v0 where G is the discounted return and v0 is the value function of the initial state (before learning)
    '''
    G = 0.
    env.reset_to_start()
    v0 = v[env.get_state()]
    for state in range(max_steps):
        if env.is_absorbing():
            break
        s1 = env.get_state()
        r = env.Act(pi[s1])
        G += r * pow(gamma, state)
        s2 = env.get_state()
        td_update(v, s1, r, s2, env.is_absorbing(), alpha, gamma)
    return G, v0


    #TODO implement the agent interacting with the environment for one episode
    # episode ends when max_steps have been completed
    # episode ends when env is in the absorbing state
    # Learning should be done online (after every step)
    # return the discounted sum of rewards G, and the value function's estimate from the initial state v0
    # the value function estimate should be before any learn takes place in this episode


def td_learning(env, pi, gamma, alpha, episodes=200, plot=True):
    '''
    Evaluates the policy pi in the environment by estimating the value function
    with TD updates  
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S|, representing the policy 
    :param gamma: discount factor
    :param alpha: learning rate
    :param episodes: number of episodes to use in evaluating the policy
    :param plot: boolean for if a plot should be generated for returns and estimates
    :return: Two lists containing the returns for each episode and the value function estimates, also returns the value function
    '''
    returns, estimates = [], []
    v = np.zeros(env.num_states)
    for state in range(episodes):
        G, v0 = td_episode(env, pi, v, gamma, alpha)
        returns.append(G)
        estimates.append(v0)
    # TODO Implement the td learning for every episode
    # value function should start at 0 for all states
    # return the list of returns, and list of estimates for all episodes
    # also return the value function v

    if plot:
        fig = plt.figure()
        plt.title("TD Learning with $\gamma={0}$ and $\\alpha={1}$".format(gamma, alpha))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(-4, 1)
        plt.plot(returns)
        plt.plot(estimates)
        plt.legend(['Returns', 'Estimate'])
        pp = PdfPages('./plots/tdplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()
    print("TDreturns\n", returns)
    print("TDestimates\n", estimates)
    print("TDv\n", v)
    return returns, estimates, v

def egreedy(q, s, eps):
   '''
   Epsilon greedy action selection for a discrete Q function.
   :param q: numpy array of size |S|X|A| representing the state action value look up table
   :param s: the current state to get an action (an integer)
   :param eps: the epsilon parameter to randomly select an action
   :return: an integer representing the action
   '''

   # TODO implement epsilon greedy action selection
   if(np.random.rand() < eps):
       return np.random.choice(range(len(q[s])))
   else:
       return np.argmax(q[s])


def q_update(q, s1, a, r, s2, terminal, alpha, gamma):
   '''
   Performs the Q learning update rule for a (s,a,r,s') transition.
   Updates to the Q values should be done inplace
   :param q: numpy array of size |S|x|A| representing the state action value table
   :param s1: current state
   :param a: action taken
   :param r: reward observed
   :param s2: next state
   :param terminal: bool for if the episode ended
   :param alpha: learning rate
   :param gamma: discount factor
   :return: None
   '''

   # TODO implement Q learning update rule
   # update should be done inplace (not returned)
   if terminal:
       q[s1][a] = (q[s1][a]) + (alpha * (r - q[s1][a]))
   else:
       max = np.argmax(q[s2])
       q[s1][a] = (q[s1][a]) + (alpha * (r + (gamma * q[s2][max]) - q[s1][a]))
       pass

def q_episode(env, q, eps, gamma, alpha, max_steps=1000):
   '''
   Agent interacts with the environment for an episode update the state action value function
   online according to the Q learning update rule. Actions are taken with an epsilon greedy policy
   :param env: environment object (GridWorld)
   :param q: numpy array of size |S|x|A| for state action value function
   :param eps: epsilon greedy parameter
   :param gamma: discount factor
   :param alpha: learning rate
   :param max_steps: maximum number of steps to interact with the environment
   :return: two floats: G, q0 which are the discounted return and the estimate of the return from the initial state
   '''
   env.reset_to_start()
   q0 = np.max(q[env.get_state()])
   G = 0
   for state in range(max_steps):
       if env.is_absorbing():
           break
       s1 = env.get_state()
       a = egreedy(q, s1, eps)
       r = env.Act(a)
       s2 = env.get_state()
       G += r * pow(gamma, state)
       q_update(q, s1, a, r, s2, env.is_absorbing(), alpha, gamma)


   # TODO implement agent interaction for q learning with epsilon greedy action selection
   # Return G the discounted some of rewards and q0 the estimate of G from the initial state

   return G, q0

def q_learning(env, eps, gamma, alpha, episodes=200, plot=True):
   '''
   Learns a policy by estimating the state action values through interactions
   with the environment.
   :param env: environment object (GridWorld)
   :param eps: epsilon greedy action selection parameter
   :param gamma: discount factor
   :param alpha: learning rate
   :param episodes: number of episodes to learn
   :param plot: boolean for if a plot should be generated returns and estimates
   :return: Two lists containing the returns for each episode and the action value function estimates of the return, also returns the Q table
   '''
   returns, estimates = [], []
   q = np.zeros((env.num_states, env.num_actions))
   # TODO implement Q learning over episodes
   # return the returns and estimates for each episode and the Q table
   for state in range(episodes):
       G, q0 = q_episode(env, q, eps, gamma, alpha)
       returns.append(G)
       estimates.append(q0)

   if plot:
       fig = plt.figure()
       plt.title("Q Learning with $\gamma={0}$, $\epsilon={1}$, and $\\alpha={2}$".format(gamma, eps, alpha))
       plt.xlabel("Episode")
       plt.ylabel("Return")
       plt.ylim(-4, 1)
       plt.plot(returns)
       plt.plot(estimates)
       plt.legend(['Returns', 'Estimate'])

       pp = PdfPages('./plots/qplot.pdf')
       pp.savefig(fig)
       plt.close()
       pp.close()

   print("Qreturns\n", returns)
   print("Qestimate\n", estimates)
   print("Qq\n", q)
   return returns, estimates, q


if __name__ == '__main__':
    env = GridWorld()
    mdp = GridWorld_MDP()
    U, pi, Ustart = policy_iteration(mdp, plot=True)
    vret, vest, v = td_learning(env, pi, gamma=1., alpha=0.1, episodes=2000, plot=True)
    qret, qest, q = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
    for num in q:
        print(max(num), list(num).index(max(list(num))))
