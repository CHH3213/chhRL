# DQN
DQN 使用深度神经网络近似拟合状态动作值函数Q(s,a).
![DQN](https://img-blog.csdnimg.cn/9a7ff4f408da4ae78fa8cbd82db54cac.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ0hIMzIxMw==,size_20,color_FFFFFF,t_70,g_se,x_16)
上图就是一般的 Deep Q-network(DQN) 的算法。

## 一般性原理
- 这个算法是这样的。初始化的时候，你初始化 2 个网络：Q 和 $\hat{Q}$，一开始这个目标 Q 网络，跟原来的 Q 网络是一样的。在每一个 episode，你拿你的演员去跟环境做互动，在每一次互动的过程中，你都会得到一个状态 $s_t$，然后你会采取某一个动作 $a_t$。怎么知道采取哪一个动作 $a_t$ 呢？就是根据你现在的 Q-function。但是要有探索的机制。比如说你用 Boltzmann 探索或是 Epsilon Greedy 的探索。那接下来你得到奖励 $r_t$，然后跳到状态 $s_{t+1}$。所以现在收集到一笔数据，这笔数据是 ($s_t$, $a_t$ ,$r_t$, $s_{t+1}$)。这笔数据放入 buffer 里面去。如果 buffer 满的话， 就再把一些旧的数据丢掉。接下来再从你的 buffer 里面去采样数据，那你采样到的是$(s_{i}, a_{i}, r_{i}, s_{i+1})$。这笔数据跟你刚放进去的不一定是同一笔，你可能抽到一个旧的。要注意的是，你采样出来的是一个 batch 的数据。接下来就是计算你的目标。假设采样出这么一笔数据。根据这笔数据去算你的目标。你的目标是什么呢？目标记得要用目标网络 $\hat{Q}$ 来算。目标是：
  $$y=r_{i}+\max _{a} \hat{Q}\left(s_{i+1}, a\right) $$
- 其中 a 就是让$\hat{Q}$的值最大的 a。接下来我们要更新 Q 值，那就把它当作一个回归问题，希望 $Q(s_i,a_i)$ 跟你的目标越接近越好。然后假设已经更新了某一个数量的次，比如说 C 次，设 C = 100， 那你就把 $\hat{Q}$设成 Q，这就是 DQN。



## Navie_DQN
没有target network的最朴素的DQN算法。
## Nature_DQN
就是标准的DQN算法，使用了target network。算法伪代码即上图所示。
## DQN_CNN
使用CNN网络而不是用MLP网络来解决gym环境的问题。
实现的算法依旧是Nature DQN。

## Double_DQN
Double DQN 算法实现
## Dueling_DQN



