# Policy and Value Iteration using Dynamic Programming
动态规划(Dynamic Programming，DP)适合解决满足如下两个性质的问题：
   - 最优子结构(`optimal substructure`)。最优子结构意味着，我们的问题可以拆分成一个个的小问题，通过解决这个小问题，最后，我们能够通过组合小问题的答案，得到大问题的答案，即最优的解。
   - 重叠子问题(`Overlapping subproblems`)。重叠子问题意味着，子问题出现多次，并且子问题的解决方案能够被重复使用。

MDP 是满足动态规划的要求的：

  - 在` Bellman equation` 里面，我们可以把它分解成一个递归的结构。当我们把它分解成一个递归的结构的时候，如果我们的子问题子状态能得到一个值，那么它的未来状态因为跟子状态是直接相连的，那我们也可以继续推算出来。
  - 价值函数就可以储存并重用它的最佳的解。

动态规划应用于 MDP 的规划问题(planning)而不是学习问题(learning)，我们必须**对环境是完全已知的(Model-Based)**，才能做动态规划，直观的说，就是**要知道状态转移概率和对应的奖励才行**.

**动态规划能够完成预测问题和控制问题的求解**，是解 MDP prediction 和 control 一个非常有效的方式。


## Policy iteration
Policy iteration 由两个步骤组成：`policy evaluation `和 `policy improvement`。
- 第一个步骤是 `policy evaluation`，在优化 policy $\pi$过程中会得到一个最新的 policy。我们先保证这个 policy 不变，然后去估计它出来的这个价值。给定当前的 `policy function` 来估计这个 价值函数。
- 第二个步骤是` policy improvement`，得到 价值 函数过后，可以进一步推算出它的 Q 函数，然后直接在 Q 函数上面取极大化，通过在Q 函数上面做贪心的搜索来进一步改进它的策略。
- 这两个步骤一直在迭代进行，所以在 `policy iteration` 里面，在初始化的时候，有一个初始化的 $V$ 和 $\pi$ 在`evaluation `和 `improvement`之间迭代。

![在这里插入图片描述](https://img-blog.csdnimg.cn/8e7acd06fffd44009872b755675691c1.png)
## Value iteration
- `Value iteration` 就是把 `Bellman Optimality Equation` 当成一个 `update rule` 来进行，如下式所示：
$$
v(s) \leftarrow \max _{a \in \mathcal{A}}\left(R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v\left(s^{\prime}\right)\right)
$$

​	上面这个等式只有当整个 MDP 已经到达最佳的状态时才满足。但这里可以把它转换成backup 的等式。我们不停地去迭代` Bellman Optimality Equation`，到了最后，它能逐渐趋向于最佳的策略，这是 value iteration 算法的精髓。

- 为了得到最佳的 $v^*$ ，对于每个状态的 $v^*$，通过` Bellman Optimality Equation `进行迭代，迭代了很多次之后，它就会收敛。



## 赌徒问题
来自[Sutton的书](http://incompleteideas.net/book/RLbook2018.pdf
)
赌徒有机会对一系列硬币翻转的结果进行赌注。
如果硬币正面，他就赢得了尽可能多的钱，因为他赌注了这一面;
如果是反面，他会失去他的赌注。当赌徒赢得100美元的目标时，游戏结束，或者输掉了所有钱。
在每次翻转时，赌徒必须决定下注哪一面，以整数金额下注。
状态是赌徒的资本，S∈{1,2，。 。 。 ，99}。动作是赌注，a {0,1，。 。 。 ，min（s，100 - s）}。当赌徒赢了，则奖励+1，否则奖励为0。
状态值函数然后给出每个状态获胜的概率。让P_H表示硬币正面朝上的概率。如果已知P_H，则已知整个问题，例如，可以通过价值迭代来解决。


