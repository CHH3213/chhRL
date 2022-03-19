## Q-learning
具体参考[表格型方法](https://blog.csdn.net/weixin_42301220/article/details/123088029)
- Q-learning 有两种 policy：behavior policy 和 target policy。

- Target policy 直接在 Q-table 上取 greedy，就取它下一步能得到的所有状态。
- Behavior policy 可以是一个随机的 policy，但我们采取 e-greedy，让 behavior policy 不至于是完全随机的，它是基于 Q-table 逐渐改进的。

- 我们可以构造 Q-learning target，**Q-learning 的 next action 都是通过 arg max 操作来选出来的**。

- 接着我们可以把 Q-learning 更新写成增量学习的形式，TD target 就变成 max 的值

## 算法伪代码
![在这里插入图片描述](https://img-blog.csdnimg.cn/95d2e059c2184ecb8272344669b375be.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ0hIMzIxMw==,size_20,color_FFFFFF,t_70,g_se,x_16)