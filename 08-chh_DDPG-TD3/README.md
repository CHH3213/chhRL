## Deep Deterministic Policy Gradient (DDPG)

论文：
http://proceedings.mlr.press/v32/silver14.pdf

https://arxiv.org/abs/1509.02971


![在这里插入图片描述](https://img-blog.csdnimg.cn/129e4a87c7b64991bebe6e849b94e5ed.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/d66233eccf9344b787754841c95be638.png)



  -  DQN 的最佳策略是想要学出一个很好的 Q 网络，学好这个网络之后，我们希望选取的那个动作使你的 Q 值最大。

 -   DDPG 的目的也是为了求解让 Q 值最大的那个 action。
       - Actor 只是为了迎合评委的打分而已，所以用来优化策略网络的梯度就是要最大化这个 Q 值，所以构造的 loss 函数就是让 Q 取一个负号。
         - 我们写代码的时候就是把这个 loss 函数扔到优化器里面，它就会自动最小化 loss，也就是最大化 Q。

这里要注意，除了策略网络要做优化，DDPG 还有一个 Q 网络也要优化。

   * 评委一开始也不知道怎么评分，它也是在一步一步的学习当中，慢慢地去给出准确的打分。

   - 优化 Q 网络的方法其实跟 DQN 优化 Q 网络的方法一样，我们用真实的 reward r 和下一步的 Q 即 Q' 来去拟合未来的收益 Q_target。

- 然后让 Q 网络的输出去逼近这个 Q_target。
     -  所以构造的 loss function 就是直接求这两个值的均方差。
     -  构造好 loss 后，通过优化器，让它自动去最小化 loss。

![在这里插入图片描述](https://img-blog.csdnimg.cn/4210b5a76e4840b394ab9894283138cd.png)

### 代码运行
- 运行main_ddpg.py,选择算法（ddpg或者td3），开始训练
    ```shell
    python main_ddpg.py --algo ddpg --train 
    ```
- 如果想要接着上次的训练，则再加上`--restore`
    ```shell
    python main_ddpg.py --algo ddpg --train --restore 
    ```
- 测试
    ```shell
    python main_ddpg.py --algo ddpg 
    ```

## Twin Delayed Deep Deterministic Policy Gradients (TD3)

论文：https://arxiv.org/abs/1802.09477

![在这里插入图片描述](https://img-blog.csdnimg.cn/574f167561be49cb84f1f0a83205759c.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/80767b6c99ea4509aa77cf544ba36d28.png)




### 代码运行
- 运行main_td3.py,选择算法（ddpg或者td3），开始训练
    ```shell
    python main_td3.py --algo td3 --train 
    ```
- 如果想要接着上次的训练，则再加上`--restore`
    ```shell
    python main_td3.py --algo td3 --train --restore 
    ```
- 测试
    ```shell
    python main_td3.py --algo td3 
    ```

效果较ddpg不如，后续再调试。
