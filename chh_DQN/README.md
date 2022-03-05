# DQN
- DQN 使用深度神经网络近似拟合状态动作值函数Q(s,a).

- 在DQN中，网络的输入是 ![[公式]](https://www.zhihu.com/equation?tex=s+) ，输出是一个向量 ![[公式]](https://www.zhihu.com/equation?tex=%28Q%28s%2Ca_1%29%2CQ%28s%2Ca_2%29%2C%5Ccdots+%2CQ%28s%2Ca_m%29%29)

- 博客参考：https://blog.csdn.net/weixin_42301220/article/details/123221485#t9



## 1. Navie_DQN
![在这里插入图片描述](https://img-blog.csdnimg.cn/cbcb947b5c8f4fb8b8fc835ae9008937.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ0hIMzIxMw==,size_20,color_FFFFFF,t_70,g_se,x_16)

没有target network的最朴素的DQN算法。

## 2. Nature_DQN
![DQN](https://img-blog.csdnimg.cn/9a7ff4f408da4ae78fa8cbd82db54cac.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ0hIMzIxMw==,size_20,color_FFFFFF,t_70,g_se,x_16)

就是标准的DQN算法，使用了target network。算法伪代码即上图所示。

- 这个算法是这样的。初始化的时候，你初始化 2 个网络：Q 和 $\hat{Q}$，一开始这个目标 Q 网络，跟原来的 Q 网络是一样的。在每一个 episode，你拿你的演员去跟环境做互动，在每一次互动的过程中，你都会得到一个状态 $s_t$，然后你会采取某一个动作 $a_t$。怎么知道采取哪一个动作 $a_t$ 呢？就是根据你现在的 Q-function。但是要有探索的机制。比如说你用 Boltzmann 探索或是 Epsilon Greedy 的探索。那接下来你得到奖励 $r_t$，然后跳到状态 $s_{t+1}$。所以现在收集到一笔数据，这笔数据是 ($s_t$, $a_t$ ,$r_t$, $s_{t+1}$)。这笔数据放入 buffer 里面去。如果 buffer 满的话， 就再把一些旧的数据丢掉。接下来再从你的 buffer 里面去采样数据，那你采样到的是$(s_{i}, a_{i}, r_{i}, s_{i+1})$。这笔数据跟你刚放进去的不一定是同一笔，你可能抽到一个旧的。要注意的是，你采样出来的是一个 batch 的数据。接下来就是计算你的目标。假设采样出这么一笔数据。根据这笔数据去算你的目标。你的目标是什么呢？目标记得要用目标网络 $\hat{Q}$ 来算。目标是：
  $$y=r_{i}+\max _{a} \hat{Q}\left(s_{i+1}, a\right) $$
- 其中 a 就是让$\hat{Q}$的值最大的 a。接下来我们要更新 Q 值，那就把它当作一个回归问题，希望 $Q(s_i,a_i)$ 跟你的目标越接近越好。然后假设已经更新了某一个数量的次，比如说 C 次，设 C = 100， 那你就把 $\hat{Q}$设成 Q，这就是 DQN。



## 3. [Double_DQN](https://arxiv.org/pdf/1509.06461.pdf)

- 在 Double DQN 里面，选动作的 Q-function 跟算目标Q值的 Q-function 不是同一个。在在 Double DQN 里面，有两个 Q-network：

  第一个 Q-network Q 决定哪一个动作的 Q 值最大（把所有的 a 带入 Q 中，看看哪一个 Q 值最大）。
  决定你的动作以后， Q 值是用 Q'算出来的。

- 假设我们有两个 Q-function，假设第一个 Q-function 高估了它现在选出来的动作 a，只要第二个 Q-function Q'没有高估这个动作 a 的值，那算出来的就还是正常的值。假设 Q'  高估了某一个动作的值，那也没差，因为只要前面这个 Q 不要选那个动作出来就没事了，这个就是 Double DQN 神奇的地方。

- Double DQN 相较于原来的 DQN 的更改是最少的，它几乎没有增加任何的运算量，连新的网络都不用，因为原来就有两个网络了。你唯一要做的事情只有，原本在找 Q 值最大的 a 的时候，你是用目标网络来算，现在改成用另外一个会更新的 Q-network 来算，然后把a带入目标网络得到目标Q值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/9e9dfe39a44a4a70afd4ea498a6f4291.png)

## 4. [Dueling_DQN](https://arxiv.org/abs/1511.06581v3)

- Dueling DQN相较于原来的 DQN，唯一的差别是改了网络的架构。

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/94ebf9bc547b4ddc84d90166641cd9b4.png)

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/b30ae98f47a844b283cc3456622f82f2.png)

实际使用的公式为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/0da11c6411794dd79e2ae9867e7c692c.png)



### 5. [DQN with Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

![在这里插入图片描述](https://img-blog.csdnimg.cn/c2d7edb57ca1407b8516f57cc540a025.png)


- 原来在 `sample` 数据去训练 `Q-network` 的时候，我们是均匀地从 `experience buffer` 里面去 `sample` 数据。那这样不见得是最好的， 因为也许有一些数据比较重要。假设有一些数据，之前有 sample 过，发现这些数据的 **TD error 特别大**（TD error 就是网络的输出跟目标之间的差距），**那这些数据代表在训练网络的时候， 训练是比较不好的**。既然比较训练不好， 那就应该给它比较大的概率被 `sample` 到，即给它 `priority`，这样在训练的时候才会多考虑那些训练不好的训练数据。实际上在做 `prioritized experience replay` 的时候，你不仅会更改 `sampling` 的 process，你还会更改更新参数的方法。所以 **`prioritized experience replay` 不仅改变了 sample 数据的分布，还改变了训练过程。**
- 具体讲解可参考：https://blog.csdn.net/hehedadaq/article/details/100127962



### 6. [Noisy_DQN](https://arxiv.org/abs/1706.01905)

https://arxiv.org/abs/1706.10295

https://arxiv.org/abs/1706.01905

DQN with noisy net的实现。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a9fec14266a04fa8b5d566be5e2871ca.png)

- `Epsilon Greedy` 的探索是在动作的空间上面加噪声，但是有一个更好的方法叫做`Noisy Net`，**它是在参数的空间上面加噪声。**
- Noisy Net 的意思是说，每一次在一个 episode 开始的时候，在跟环境互动的时候，在Q网络的每一个参数上面加上一个高斯噪声(Gaussian noise)，那就把原来的 Q-function 变成 $\tilde{Q}$ ，就得到一个新的网络叫做 $\tilde{Q}$。
- 这边要注意在每个 episode 开始的时候，开始跟环境互动之前，我们就 sample 网络。接下来用这个固定住的 noisy网络去玩这个游戏，**直到游戏结束，才重新再去 sample 新的噪声**。





### 7. [N_Step_DQN](https://arxiv.org/abs/1901.07510#:~:text=Multi-step%20methods%20such%20as%20Retrace%20%28%29%20and%20-step,to%20draw%20statistically%20significant%20conclusions%20about%20their%20performance.)

*待实现。*

balance MC 跟 TD（即多步TD）。MC 跟 TD 的方法各自有优劣，怎么在 MC 跟 TD 里面取得一个平衡呢？我们可以不要只存一个步骤的数据，我们存 N 个步骤的数据。



### 8. [Distributional_DQN](https://arxiv.org/pdf/1707.06887.pdf)

- Distributional DQN，就是把DQN中的**value function**换成了**value distribution**。

- 原来的DQN中的值函数是 ![[公式]](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29) ，它是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E%7Bn%7D+%5Ctimes+%5Cmathbb+R%5E%7Bm%7D+%5Crightarrow%5Cmathbb++R) 的函数，也就和说，它接受一个 ![[公式]](https://www.zhihu.com/equation?tex=s%2Ca) ，输出一个实数，这个实数就是这个状态动作对 ![[公式]](https://www.zhihu.com/equation?tex=a%2Cs) 的评估。

- Distributional DQN中的**值分布**，它接受一个 ![[公式]](https://www.zhihu.com/equation?tex=s%2Ca) ，输出一个**分布**，这个分布描绘了状态动作对 ![[公式]](https://www.zhihu.com/equation?tex=a%2Cs) 的**所有取值的可能性**。

- 简单来说，**值函数可以看作值分布的期望**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/17a80fb890d34bd79029e633f8a15105.png)

表示分布当然有很多方法，高斯分布这种参数化的方法，因为它是**单峰**的，所以不使用。

我们希望能实现**多个峰值**的值分布，从而学习到更好的结果。有作者们提出了一个叫做C51的算法，它用51个等间距的atoms描绘一个分布。具体可参考博客：[Distributional DQN: C51](https://zhuanlan.zhihu.com/p/137935717#:~:text=%E6%89%80%E8%B0%93%E7%9A%84Distributional%20DQN%EF%BC%8C%E5%B0%B1%E6%98%AF%E6%8A%8A%E4%BC%A0%E7%BB%9FDQN%E4%B8%AD%E7%9A%84%20value%20function%20%E6%8D%A2%E6%88%90%E4%BA%86%20value,distribution%20%E3%80%82%20%E5%8E%9F%E6%9D%A5%E7%9A%84DQN%E4%B8%AD%E7%9A%84%E5%80%BC%E5%87%BD%E6%95%B0%E6%98%AF%20%EF%BC%8C%E5%AE%83%E6%98%AF%20%E7%9A%84%E5%87%BD%E6%95%B0%EF%BC%8C%E4%B9%9F%E5%B0%B1%E5%92%8C%E8%AF%B4%EF%BC%8C%E5%AE%83%E6%8E%A5%E5%8F%97%E4%B8%80%E4%B8%AA%20%EF%BC%8C%E8%BE%93%E5%87%BA%E4%B8%80%E4%B8%AA%E5%AE%9E%E6%95%B0%EF%BC%8C%E8%BF%99%E4%B8%AA%E5%AE%9E%E6%95%B0%E5%B0%B1%E6%98%AF%E8%BF%99%E4%B8%AA%E7%8A%B6%E6%80%81%E5%8A%A8%E4%BD%9C%E5%AF%B9%20%E7%9A%84%E8%AF%84%E4%BC%B0%E3%80%82)



### 9. Rainbow

把刚才所有的方法(Nature DQN,Double_DQN,Dueling_DQN,Prioritized Experience Replay,Noisy_DQN,N_Step_DQN, Distributional_DQN)都综合起来就变成 rainbow 。因为刚才每一个方法，就是有一种自己的颜色，把所有的颜色通通都合起来，就变成 rainbow，它把原来的 DQN 也算是一种方法，故有 7 色。



> N_Step_DQN暂未实现，所以先不加入改trick。Noisy_DQN加入后代码部分有问题，还为解决，所以先不加入。Distributional_DQN比较复杂，还不是很懂，所以暂时不加入。目前实现的Rainbow包含了：Nature DQN,Double_DQN,Dueling_DQN,Prioritized Experience Replay 等4种trick。
