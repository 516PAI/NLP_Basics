# 决策树、随机森林、XGBoost、LightGBM

## 决策树

### 简单理解

简单理解决策树，其实就是一堆if、else拼凑，直到做出分类结果

![image-20230620172126719](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230620172126719.png)

在决策树中，除了叶子结点和输入，其它的每个结点都代表了某个特征。

老鹰：（有羽毛，会飞）

熊：（没有羽毛，没有鳍）



### 算法

决策树有个思路：**优先判断重要的特征**

如何判断特征的重要性？答：我需要做判断的次数越少越好（信息纯度高）

> 信息纯度
>
> > 与信息熵成反比，信息量越大，信息越杂乱，纯度越低；信息量越小，信息越规整，纯度越高。

如何提高信息纯度？

当一个分支下所有样本都属于同一个类，纯度最高

当一个分支下正负类比例为1：1，纯度最低

那实际上就是要找到一种度量函数，需要它来度量纯度

![image-20230620204125027](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230620204125027.png)

> （1）当决策树是二叉树时，a取值只有0，1；当决策树是多叉树时，a取值有多种可能。
>
> （2）canStop的条件可以是：1. 当所有特征都已经用完时，无法再分；2. 当前集合为空，无法划分



**scoreIfSplitBy**

- 信息增益

  - $$
    IG(D, A) = H(D) - H(D|A)
    $$

- 信息增益率

  - $$
    IGR(D, A) = IG(D, A) / H(A)
    $$

- Gini系数

  - $$
    Gini(P) = \sum^K_{k=1}P_k * (1-P_k) \\
    Gini(P) = (1 - \sum^K_{k=1}P^2_k)
    $$



**什么是信息量**

假设$X=x_i$的概率为$p(x_i)$，那么$X=x_i$的信息量可以表示为$I(X=x_i)$，而$I(X=x_i) = -log_2p(x_i)$。

直观理解，即当小概率事件发生时，信息量越大；当大概率事件发生时，信息量越小。

**熵**

熵是对事件不确定性的度量。

**信息熵**

信息熵是对事件不确定性程度的度量（平均信息量）。
$$
H(D) = \sum^n_{i=1}p(x_i)I(x_i) = - \sum^n_{i=1}p(x_i)log_2p(x_i)
$$

> 假设我论文中稿概率是90%，拒稿概率是10%，那我中稿与否的不确定性程度如何度量呢？

$$
H(投稿情况) = -(0.9 * log_2 0.9 + 0.1 * log_2 0.1) \\ 
= - (0.9 * (- 0.152) + 0.1 * (-3.322)) \\
= -(-0.1368-0.3322) = 0.469
$$

中稿概率很高，所以基本能确定我能中稿，所以不确定性程度比较低。实际上，该计算存在对称性，假如拒稿概率90%，中概率10%，信息熵也是0.469。

> 我投掷一颗均匀的硬币，它是正反面的不确定性程度如何度量呢？

$$
H(投掷硬币) = - (0.5 * log_2 0.5 + 0.5 * log_2 0.5) \\
=1
$$

> 假设我能瘦下来的概率是100%，瘦不下来的概率是0%，那我能不能瘦下来的不确定性程度如何度量？

$$
H(我能不能瘦) = -(1 * log_2 1 + 0 * log_2 0) \\
= 0
$$

**条件熵**
$$
H(Y|X) = \sum_xp(x)H(Y|X=x)
$$
已知条件X情况下，Y的信息熵。



#### ID3（Information Divergence）

优先选择**信息增益**最大的特征

>  信息增益是信息熵与在特征A下的条件熵H(D|A)之差
>
> $IG(D, A) = H(D) - H(D|A)$



**划分思想**

1. 首先计算未分裂前当前集合D的信息熵H(D)
2. 计算当前集合D对所包含的所有特征A的条件熵H(D|A)
3. 然后计算每个特征的信息增益IG(D|A)
4. 选取信息增益最大的特征A作为决策树进行分裂



**例子**

| 样本ID | 季节       | 时间已过 8 点 | 风力情况 | 要不要赖床 |
| ------ | ---------- | ------------- | -------- | ---------- |
| 1      | spring     | no            | breeze   | yes        |
| 2      | **winter** | no            | no wind  | **yes**    |
| 3      | autumn     | yes           | breeze   | yes        |
| 4      | **winter** | no            | no wind  | yes        |
| 5      | summer     | no            | breeze   | yes        |
| 6      | **winter** | yes           | breeze   | **yes**    |
| 7      | **winter** | no            | gale     | **yes**    |
| 8      | **winter** | no            | no wind  | **yes**    |
| 9      | spring     | yes           | no wind  | no         |
| 10     | summer     | yes           | gale     | no         |
| 11     | summer     | no            | gale     | no         |
| 12     | autumn     | yes           | breeze   | no         |

- 第一次做划分前的熵

$P(赖床)=\frac{2}{3}$

$P(不赖床)=\frac{1}{3}$

$H(D)=-(\frac{2}{3}*log\frac{2}{3} + \frac{1}{3}*log\frac{1}{3})=0.636$

H(D)=-(\frac{2}{3}*log\frac{2}{3} + \frac{1}{3}*log\frac{1}{3})=0.636

- 条件熵

$H(D|breeze)=P(X=Breeze)H(D|X=Breeze)=-\frac{5}{12}*(\frac{4}{5}log\frac{4}{5}+\frac{1}{5}log\frac{1}{5})=0.209$

$H(D|nowind)=P(X=nowind)H(D|X=nowind)=-\frac{4}{12}*(\frac{3}{4}log\frac{3}{4}+\frac{1}{4}log\frac{1}{4})=0.187$

$H(D|gale)=P(X=gale)H(D|X=gale)=-\frac{3}{12}*(\frac{2}{3}log\frac{2}{3}+\frac{1}{3}log\frac{1}{3})=0.212$

- 信息增益

$IG(D, 风力情况) = H(D) - H(D|风力情况) = 0.636 - (0.209 + 0.187 + 0.212) = 0.028$

同理得

$IG(D, 季节) = H(D) - H(D|季节) = 0.636 - 0.279 = 0.357$

$IG(D, 时间) = H(D) - H(D|时间) = 0.636 - 0.519 = 0.117$

因此，此时模型倾向于选择季节作为第一个分裂特征。



**优点**

1）考虑了特征出现与不出现的两种情况

>  讲不通，我觉得这不是一个优点

我在其它提到这个优点的博客中看到下面这个例子，但是这个例子似乎并不能说明这个“优点”，你只要给了这个特征0或1的取值，那不管什么算法都会考虑

![image-20230628104317093](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230628104317093.png)

2）使用了所有样例的统计属性，减小了对噪声的敏感度

> 每次选取信息增益最大的特征

3）容易理解，计算简单

**缺点**

1）仅考察特征对整个系统的贡献，没有具体到类别熵，只适用于全局的特征选择，无法针对单个类别做特征选择

>**什么叫无法针对单个类别做特征选择？**拿篇章关系的例子来讨论，当连接词为“After”时，基本可以确定是Temporal关系，也就是通过连接词是否为“After”就能得到一个纯度很高的子集。但是，连接词是否为“After”对其它关系类型没有任何作用。
>
>**只做全局的特征选择的一个极端情况**，就是下面第二条缺点

2）算法偏向选择分支多的特征，容易过拟合（假设唯一标识ID作为一个属性值的话，ID的信息增益会最大）

$H(D|样本ID) = 12 * (-\frac{1}{1}log\frac{1}{1}) = 0$

$IG(D, ID) = 0.636 - 0 = 0.636$



#### C4.5

为了缓解ID3过拟合分支多的特征，所以需要引入一个惩罚参数，惩罚分支过多的情况。

即特征A的信息熵的倒数$H(A)$，所以度量指标变成信息增益率：$IGR(D|A)=IG(D|A)/H(A)$

> 为啥信息熵的倒数可以惩罚分支过多的情况？2
>
> 分支过多，则每个结果的概率都很低，回到最开始的说法：“小概率事件，信息量更大”。

当ID作为分裂特征时，$H(A)=-12 * \frac{1}{12}log(\frac{1}{12}) = 2.485$

相应的，信息增益率为$IGR(D|A) = 0.636 / 2.485 = 0.256$

同理，风力的信息增益率为0.026

季节的信息增益率为0.273

时间的信息增益率为0.172

最后，C4.5算法仍然优先选择季节作为划分标准



<todo> 待补充

**连续特征离散化**

假设成年人的正常身高为140cm~210cm，那么身高这个特征就是一个连续特征。

假设有5个样本，身高特征分别为[150, 160, 190, 170, 180]，对应标签为[0, 1, 1, 0, 1]

- 首先对样本排序[150, 160, 170, 180, 190], [0, 1, 0, 1, 1]
- 选择合适条件切分数据，例如[150, 180), [180, 190]
- 对切分后的数据来计算熵的增益率，挑选增益率最大的方式来划分数据。



#### CART

Classification and Regression Trees

$Gini(P) = \sum^K_{k=1}P_k * (1-P_k) = (1 - \sum^K_{k=1}P^2_k)$

$P_k$表示样本属于k类别的概率，那不属于k类别的概率则为$1-P_k$

基于特征A划分样本集合D之后的基尼指数：

$Gini(D,A) = \frac{D_1}{D}Gini(D_1) + \frac{D_2}{D}Gini(D_2)$

CART算法是一种**二分递归分割**的算法，**最终构成的是一颗二叉树**。



**划分思想**

- 对样本集合D包含的每个特征A根据其取值构造系列二分子集
- 计算D基于该特征A的每一种取值划分所获得的二分子集的Gini指数
- 选取Gini指数最小的特征取值作为最优划分点



**例子**

季节

- $Gini(季节=春) = 1- (\frac{1}{2})^2- (\frac{1}{2})^2=0.5 $
- $Gini(季节=夏) = 1- (\frac{1}{3})^2- (\frac{2}{3})^2=0.44 $
- $Gini(季节=秋) = 1- (\frac{1}{2})^2- (\frac{1}{2})^2=0.5 $
- $Gini(季节=冬) = 1- (\frac{5}{5})^2=0 $

风力

- $Gini(风力=breeze)=1-(\frac{1}{5})^2 - (\frac{4}{5})^2 = 0.320$
- nowind: 0.375
- gale 0.44

时间

- $Gini(时间=已过8点)=1-(\frac{2}{5})^2 - (\frac{3}{5})^2 = 0.48$
- 未过8点0.245

从而，**季节=冬**时的Gini指数最小，因此用它将数据划分。

此时变成两个数据集，一个纯粹是“赖床”分类的子集，另一个是还需要处理的子集。



#### 剪枝

讲得不错的视频：

https://www.bilibili.com/video/BV1Ro4y1Q7Ha/?p=19&spm_id_from=pageDriver

- 预剪枝

分裂前判断是否会提升在验证集上的准确度，如果会降低就中止这次分裂

1. 设置树的最大深度
2. 当前结点剩余特征取值都一致，即便不属于同一类，也可以不分了
3. 设置每个叶子结点的样本数最小值
4. 判断每次分裂是否有增益，或者增益是否大于某个阈值

训练时间小，由于是基于“贪心”策略，存在欠拟合风险

- 后剪枝
  - 错误率降低剪枝 - 从下至上遍历所有非叶子节点的子树，若去掉当前子树，是否提升（需要验证集）
  - 悲观剪枝 - 自上而下（只需要训练集）
  - 最小误差剪枝 - 自下而上（只需要训练集）
  - 基于错误剪枝 - 自下而上（只需要训练集）
  - 代价复杂度剪枝

欠拟合风险小，但训练时间大

**悲观剪枝**

N(T)：样本数，L：叶子结点个数，error（Leaf$_i$）：第i个叶子结点分类错误个数

错误与非错误符合二项分布，假设p为错误个数，q为非错误个数，方差就是npq

![image-20230621144933442](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621144933442.png)

![image-20230621144353358](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621144353358.png)

 剪枝之后误判个数期望值大于剪枝前的误判的上限，所以不剪枝



**最小误差剪枝**

考虑了先验概率

![image-20230621145510717](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621145510717.png)

​     假设$Pr_k(T)=\frac{1}{K}, m = K$，这里$K$假设是类别个数，$N(T)$是总样本数，$n_k(T)$是正确样本数。

![image-20230621150633018](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621150633018.png)

![image-20230621151943401](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621151943401.png)

**基于错误剪枝**

当$\alpha=25\%$时（置信水平），$q_\alpha =0.6925$（上分为点）

![image-20230621153030982](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621153030982.png)

![image-20230621154458688](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621154458688.png)

**代价复杂度剪枝（CART剪枝算法）**

$C_\alpha = C(T) + \alpha|T|$

![image-20230621170104464](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621170104464.png)





$T_t$剪枝的话，变成一个叶子结点

①当$\alpha$趋于0，也就是不关心复杂度带来的惩罚，通常得到最完整的树，拟合能力强，泛化差

②当$\alpha$趋于正无穷，最后剪枝到只剩根结点，泛化能力强，拟合能力极差

③应当找到一个合适的α值，使得拟合能力和泛化能力综合最优，即找到某个临界点，剪枝与不剪枝的代价都相等

![image-20230628112810213](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230628112810213.png)

![image-20230628113944786](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230628113944786.png)

例子：https://zhuanlan.zhihu.com/p/548190779

<todo>



### 代码

![image-20230621132945429](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230621132945429.png)

```python
model = sklearn.tree.DecisionTreeClassifier(
	criterion=’gini’, 
	splitter=’best’, 
	max_depth=None, 
	min_samples_split=2, 
	min_samples_leaf=1, 
	min_weight_fraction_leaf=0.0, 
	max_features=None, 
	random_state=None, 
	max_leaf_nodes=None, 
	min_impurity_decrease=0.0, 
	min_impurity_split=None, 
	class_weight=None, 
	presort=False)
```

```python
#导入所需要的包
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import classification_report

# 加载模型
model = DecisionTreeClassifier()
 
# 训练模型
model.fit(X_train,y_train)
# 预测值
y_pred = model.predict(X_test)
 
'''
评估指标
'''
# 求出预测和真实一样的数目
true = np.sum(y_pred == y_test )
print('预测对的结果数目为：', true)
print('预测错的的结果数目为：', y_test.shape[0]-true)
# 评估指标
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
print('预测数据的准确率为： {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
print('预测数据的精确率为：{:.4}%'.format(
      precision_score(y_test,y_pred)*100))
print('预测数据的召回率为：{:.4}%'.format(
      recall_score(y_test,y_pred)*100))
# print("训练数据的F1值为：", f1score_train)
print('预测数据的F1值为：',
      f1_score(y_test,y_pred))
print('预测数据的Cohen’s Kappa系数为：',
      cohen_kappa_score(y_test,y_pred))
# 打印分类报告
print('预测数据的分类报告为：','\n',
      classification_report(y_test,y_pred))
```





## 集成学习

### Bagging

随机森林

实际上就是训练多个决策树，然后投票得到最终的决策



### Boosting

AdaBoost、GBDT

**Boosting工作机制：**

- 首先从训练集**用初始权重训**练出一个弱学习器1；
- 根据弱学习的学习**误差率表现来更新训练样本的权重**，使之前弱学习器1学习误差率高的训练样本点的权重变高，即让误差率高的点在后面的弱学习器2中得到更多的重视；
- 然后**基于调整权重后的训练集来训**练弱学习器2；
- 如此重复进行，直到弱学习器数达到事先指定的数目T；
- 最终将这T个弱学习器通过集成策略进行整合，得到最终的强学习器。



**问题**

1）在每一轮如何改变训练数据的权值或概率分布？

通过提高那些在前一轮被弱分类器分错样例的权值，减小前一轮分对样例的权值，来使得分类器对误分的数据有较好的效果。

2）通过什么方式来组合弱分类器？

通过加法模型将弱分类器进行线性组合，比如：AdaBoost（Adaptive boosting）算法：刚开始训练时对每一个训练样例赋相等的权重，然后用该算法对训练集训练t轮，每次训练后，对训练失败的训练例赋以较大的权重，也就是让学习算法在每次学习以后更注意学错的样本，从而得到多个预测函数，将每一步生成的模型叠加得到最终模型。GBDT（Gradient Boost Decision Tree)，每一次的计算是为了减少上一次的残差，GBDT在残差减少（负梯度）的方向上建立一个新的模型。



### Stacking

Stacking(堆叠各种各样的分类器（KNN,SVM,RF等等），分阶段操作：

- 第一阶段输入数据特征得出各自结果

- 第二阶段再用前一阶段结果训练得到分类结果

- 训练一个模型用于组合其他各个模型： 将训练好的所有基模型对训练集进行预测，第j个基模型对第i个训练样本的**预测值将作为新的训练集中第i个样本的第j个特征值**

- 最后基于新的训练集进行训练
- 同理，预测的过程也要先经过所有基模型的预测形成新的测试集，最后再对测试集进行预测



**Stacking工作机制：**

- 首先先训练多个不同的模型；
- 然后把之前训练的各个模型的输出为输入来训练一个模型，以得到一个最终的输出。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

data = np.loadtxt("../data/wine.data")
X = data[:, 1:]
y = data[:, 0:1]
X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), train_size=0.8, random_state=0)

#定义基分类器
clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

#定义堆叠集
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True)

#对每一个模型分别进行评价
for model in [clf1, clf2, clf3, lr, sclf]:
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)
    print(model.__class__.__name__,',test accuracy:', accuracy_score(y_test, y_test_hat))
```





 **Bagging，Boosting二者之间的区别**

1）样本选择上：

Bagging：训练集是在原始集中**有放回选取的**，从原始集中选出的各轮训练集之间是独立的。

Boosting：**每一轮的训练集不变**，只是训练集中每个样例在分类器中的**权重发生变化**。而权值是根据上一轮的分类结果进行调整。

2）样例权重：

Bagging：使用均匀取样，**每个样例的权重相等**

Boosting：根据错误率不断**调整样例的权值**，错误率越大则权重越大。

3）预测函数：

Bagging：所有预测**模型的权重相等**。

Boosting：**每个弱分类器都有相应的权重**，对于分类误差小的分类器会有更大的权重。

4）并行计算：

Bagging：各个预测函数**可以并行**生成。

Boosting：各个预测函数**只能顺序生成**，因为后一个模型参数需要前一轮模型的结果。



## 随机森林

决策树容易对训练数据过拟合，随机森林是解决办法之一，它本质上是许多决策树的集合。

它的思想就是每棵树都可能会拟合部分数据，但每棵树都以不同方式过拟合，对这些树取平均值就能降低过拟合。



- 确定构造的树的个数（假设10个）
- 有放回地采样来构造大小为n的数据集（[1, 2, 3, 4, 5]  --> [1, 1, 3, 4, 4])
- 针对新的数据集，每个结点考虑的是**全部特征的一个子集，而不是所有特征**，特征个数有超参决定
- 不同数据集与不同特征子集保证了每棵树的不同

>  集合中的每一棵树都是从**训练集采样出来的样本中构建的**。在树构建期间分割节点时，所选择的分割**不再是所有特征之间最好的分割**。相反，被选中的分割**是特征的随机子集之间最好的分割**。由于随机性，森林的偏向通常略有增加。但是，由于平均值，它的方差也减小，从而产生一个**整体更好的模型**。



- 集成算法，准确性高
- 引入两个随机性，缓解过拟合
- 两个随机性使得其具备抗噪能力
- 能够处理较高维度的数据，不需要人为做特征选择，对数据集的适应能力强。
- 训练快
- 可以处理缺省值

> 使用随机森林进行缺失值填补的思想：
>
>>X和y之间是有联系的，所以才能用X预测y;那么反过来，y也可以在一定程度上预测X。
>>
>>   当X中的某个特征x有缺失值时，我们将该特征看为target，y看作一个新特征（即X去除x和y组成特征向量，x作为target）；无缺失值的样本做训练集，有缺失值的样本做测试集，使用随机森林建模（可以是回归，也可是是分类），对缺失值进行预测。
>>
>>   当X中有多个特征有缺失值时，从缺失值最少的特征开始处理，此时其他缺失值用0填充；当该特征的缺失值用随机森林预测出来后，填补到原始数据中，之后继续按上述方法处理下一个缺失值。

- 针对“袋外数据”，可以在模型生成过程中取得真实误差的无偏估计

> **袋外错误率oob error（out-of-bag error）**
>
> 　　随机森林有一个重要的优点就是，没有必要对它进行交叉验证或者用一个独立的测试集来获得误差的一个无偏估计。它可以在内部进行评估，也就是说在生成的过程中就可以对误差建立一个无偏估计。
>
> 　　我们知道，在构建每棵树时，我们对训练集使用了不同的bootstrap sample（随机且有放回地抽取）。所以对于每棵树而言（假设对于第k棵树），大约有1/3的训练实例没有参与第k棵树的生成，它们称为第k棵树的oob样本。
>
> 　　而这样的采样特点就允许我们进行oob估计，它的计算方式如下：
>
> 　　（note：以样本为单位）
>
> 　　1）对每个样本，**计算它作为oob样本的树对它的分类情况**（约1/3的树）；
>
> 　　2）然后以简单**多数投票作为该样本的分类结果**；
>
> 　　3）最后**用误分个数占样本总数的比率**作为随机森林的oob误分率。

- 训练过程中，可以检测到不同特征间的互相影响
- 可以并行训练多棵树
- 随机森林是基于CART决策树构成的。



提高随机森林模型的方法

- 特征选择
  - 输入特征按照重要性从高到低排序，可以根据与输出变量的皮尔森相关系数或者由支持向量机模型得出
  - 去除与输出变量相关性很小的特征（固定其他特征，变更某个特征的重要性，来确定它的相关性）
  - 在原有特征基础上添加新的特征，可以是原有特征集的组合或划分。（例如，把周末与假期合并成节假日；把年拆成各个月份）

- 参数优化
  - n_estimators：树的数量
  - max_features：每个结点随机选择的最大特征数
  - ....

```python
#导入所需要的包
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report#评估报告

model=RandomForestClassifier()
# 训练模型
model.fit(X_train,y_train)
# 预测值
y_pred = model.predict(X_test)
 
'''
评估指标
'''
# 求出预测和真实一样的数目
true = np.sum(y_pred == y_test )
print('预测对的结果数目为：', true)
print('预测错的的结果数目为：', y_test.shape[0]-true)
# 评估指标
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
print('预测数据的准确率为： {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
print('预测数据的精确率为：{:.4}%'.format(
      precision_score(y_test,y_pred)*100))
print('预测数据的召回率为：{:.4}%'.format(
      recall_score(y_test,y_pred)*100))
# print("训练数据的F1值为：", f1score_train)
print('预测数据的F1值为：',
      f1_score(y_test,y_pred))
print('预测数据的Cohen’s Kappa系数为：',
      cohen_kappa_score(y_test,y_pred))
# 打印分类报告
print('预测数据的分类报告为：','\n',
      classification_report(y_test,y_pred))
```



## AdaBoost

https://zhuanlan.zhihu.com/p/27126737

只处理二分类问题

公式：

$G(x) = sign[f(x)] = sign[\alpha_1G_1(x) + \alpha_2G_2(x)+ ··· +\alpha_nG_n(x)]$

| 序号 | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| x    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| w    | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  |
| y    | 1    | 1    | 1    | -1   | -1   | -1   | 1    | 1    | 1    | -1   |

**步骤**

《统计机器学习》157页公式8.1。

- 取分类阈值为2.5，得到的第一个个体学习器的分类误差率最低，此时样本7、8、9分类错误，

$$
G_1(x)= \left\{
\begin{matrix}
	1, x<2.5,\\
	-1,x>2.5
\end{matrix}
\right.
$$

- 根据公式$\alpha_i = \frac{1}{2} log \frac{1-e_i}{e_i}$，计算误差系数$\alpha_1= \frac{1}{2}log\frac{1-0.3}{0.3}=\frac{1}{2}log\frac{7}{3}=0.4236$，其中$e_i=0.1+0.1+0.1=0.3$

> 误差大于0.5时，$\frac{1-e_i}{e_i} < 1$，log为负数；反之，误差小于0.5时，log为正数。
>
> 误差越小，log值越大，$\alpha$值越大
>
> > 我找的资料，他计算的时候log以e为底，应该都可以，无非数值略有不同，相对大小仍然是一致的

- 更新训练数据的权重分布
  - $w_{m+1, i}=\frac{w_{m,i}}{Z_m}exp(-\alpha_my_iGm(x_i)), i=1,2,...,N$
  - 若$y_i$和$G_m(x_i)$同号，则为1，那括号里是负数，值就小；若异号，括号里就是正数，值就大。
  - 归一化因子$Z_m=\sum^N_{i=1}w_{m,i}exp(-\alpha_my_iGm(x_i))$

| 序号 | 1       | 2       | 3       | 4       | 5       | 6       | 7       | 8       | 9       | 10      |
| ---- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| x    | 0       | 1       | 2       | 3       | 4       | 5       | 6       | 7       | 8       | 9       |
| w    | 0.07143 | 0.07143 | 0.07143 | 0.07143 | 0.07143 | 0.07143 | 0.16667 | 0.16667 | 0.16667 | 0.07143 |
| y    | 1       | 1       | 1       | -1      | -1      | -1      | 1       | 1       | 1       | -1      |

第二次会把阈值定在8.5，因为6，7，8的权重高，会优先保证6，7，8做对，这样误差$e_i=0.07143*3=0.2143$；否则，如果还选择2.5，误差$e_i=0.16667*3=0.5$。
$$
G_2(x)= \left\{
\begin{matrix}
	1, x<8.5,\\
	-1,x>8.5
\end{matrix}
\right. \ \ \ \ \ \ \ \ \ \ \ \ \  \alpha_2=0.6496
$$

$$
f_2(x) = \alpha_1G_1(x) + \alpha_2G_2(x) = 0.4236G_1(x)+0.6496G_2(x) \\ \\
f_2(x)= \left\{
\begin{matrix}
	0.4236*1 + 0.6496*1 = 1.0732, x<2.5, \\
	0.4236*(-1) + 0.6496*(1) = 0.226, 2.5<x<8.5, \\
	0.6496*(-1) + 0.4236*(-1) = - 1.0732, x>8.5
\end{matrix}
\right. \\ \\
 sign[f_2(x)] = \left\{
\begin{matrix}
	1, x < 8.5, \\
	-1, x > 8.5
\end{matrix}
\right.
$$

| 序号 | 1      | 2      | 3      | 4       | 5       | 6       | 7      | 8      | 9      | 10     |
| ---- | ------ | ------ | ------ | ------- | ------- | ------- | ------ | ------ | ------ | ------ |
| x    | 0      | 1      | 2      | 3       | 4       | 5       | 6      | 7      | 8      | 9      |
| w    | 0.0455 | 0.0455 | 0.0455 | 0.16667 | 0.16667 | 0.16667 | 0.1060 | 0.1060 | 0.1060 | 0.0455 |
| y    | 1      | 1      | 1      | -1      | -1      | -1      | 1      | 1      | 1      | -1     |



不断通过上述步骤调整，直到得到T个个体学习器，通过$\alpha_m$调整各个学习器的参数

最终在第三次分割的时候实现了0误分类
$$
G_3(x) = \left\{
\begin{matrix}
-1, x<5.5 \\
1, x>5.5
\end{matrix}
\right.
$$

$$
f_3(x) = \alpha_1G_1(x) + \alpha_2G_2(x) + \alpha_3G_3(x) = 0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x) \\ \\
f_3(x)= \left\{
\begin{matrix}
	0.3218, x<2.5, \\
	-0.5254, 2.5<x<5.5, \\
	0.9974, 5.5<x<8.5, \\
	-0.3218, x>8.5
\end{matrix}
\right. \\ \\
 sign[f_3(x)] = \left\{
\begin{matrix}
	1, x < 2.5, \\
	-1, 2.5<x<5.5, \\
	1, 5.5<x<8.5, \\
	-1, x>8.5
\end{matrix}
\right.
$$




## GBDT

Gradient Boosting Decision Tree，使用的决策树是CART回归树

> 为什么是回归树，而不是分类数？
>
> 答：因为处理的是梯度，是个连续值

GBDT使用平方误差来作为评价指标

**算法**

- **构建回归树**

在训练数据集所在的输入空间中，递归的将每个区域划分为两个子区域并决定每个子区域上的输出值，**构建二叉决策树**:

1）切分特征j，和切分点s，求解
$$
\min_{j,s}[\min_{c_1}\sum_{x_i\in R_1(j,s)}(y_i -c_1)^2+\min_{c_2}\sum_{x_i\in R_2(j,s)}(y_i-c_2)^2]
$$
遍历特征j，对特征扫描切分点s，选择使上式取值最小的(j, s)对



2）使用选择的(j, s)划分区域，得到相应的输出

$R_1(j, s)=x|x^{(j)} \leq s, R_2(j, s) = x| x^{(j)} \gt s $

$c_m = \frac{1}{N}\sum_{x_i \in R_m(j, s)}y_i, m=1,2$



3）重复上述步骤，直到满足停止条件

4）将输入空间划分为M个区域$R_1, R_2, ..., R_m$，生成决策树：

$f(x) = \sum^M_{m=1}c_mI(x \in R_m)$



- **梯度提升**

1）初始化$f_0(x) = 0$

2）对$m=1,2,...,M$：1）计算残差$r_{mi}=y_i-f_{m-1}(x_i), i=1,2,...,N$；   2）拟合残差$r_{mi}$学习一个回归树，得到$h_m(x)$； 3）更新$f_m(x)=f_{m-1}(x)+h_m(x)$。

> 这个$h_m(x)$咋求？
>
> 答：把残差当作新的标签，训一个新的决策树

3）得到回归问题提升树

$f_M(x) = \sum^M_{m=1}h_m(x)$



**残差是啥？**

看平方损失函数：

$L(y, f_{t-1}(x)+h_t(x)) = (y-f_{t-1}(x) - h_t(x))^2 = (r-h_t(x))^2$

$r = y - f_{t-1}(x)$

实际上我下一个模型能预测出这个r就行了，我把上一个模型和这个模型的结果加起来，就能得到与y最接近的值。



**残差与梯度**

如果使用其它损失函数，可能优化起来比较麻烦，所以Freidman提出了梯度提升树算法，利用损失函数的负梯度作为提升树算法中的残差的近似值。

而平方损失的负梯度实际上和上面的残差$r$是一样的：

$L(y, f(x)) = \frac{1}{2}(y-f(x_i))^2$

$-[\frac{dL(y, f(x))}{df(x_i)}] = -(-1)*(y-f(x_i)) = y - f(x_i)$



- **GBDT**

1. 初始化弱学习器

$f_0(x)=arg\min_{c}\sum^N_{i=1}L(y_i, c)$

2. 对$m=1,2,...,M$，计算残差：

$r_{mi} = -[\frac{dL(y_i, f(x_i))}{df(x_i)}]$

3. 将残差作为样本新的真实值，并将数据$(x_i, r_{im})$作为下棵树的训练数据，训出$h_m(x)$，然后得到$f_m(x)$

$r_{im} = arg \min _{r}\sum_{x_i \in R_{jm}}L(y_i, f_{m-1}(x_i)+r)$ 

4. 新的学习器

$f_m(x) = f_{m-1}(x) + \sum_{j=1}^ J r_{jm}I(x \in R_{jm})$

5. 最终学习器

$f(x) = f_M(x) = f_0(x) + \sum^M_{m=1}\sum^J_{j=1}r_{jm}I(x \in R_{jm})$

![image-20230622170103078](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230622170103078.png)



## XGBoost

- XGBoost其实是GBDT算法的一种工程化实现



看XGBoost的目标函数的定义，GBDT的残差可以理解成目标函数的一阶导（上节提过了），而XGBoost则细化到二阶导，可以更好的逼近这个残差
$$
Obj^{(t)} = \sum^n_{i=1}l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)+constant \\
泰勒: f(x+\Delta x) \simeq f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2 \\
令:g_i = \partial_{\hat{y}_i^{(t-1)}}l(y_i, \hat{y}_i^{(t-1)}), h_i=\partial^2_{\hat{y}^{(t-1)}}l(y_i, \hat{y}_i^{(t-1)}) \\
Obj^{(t)} \simeq \sum^n_{i=1}[l(y_i, \hat{y}_i^{(t-1)}) +g_if_t(x_i)+ \frac{1}{2}h_if_t^2(x_i)] + \Omega(f_t)+constant
$$


**正则项** $\Omega(f_t)$

$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum^T_{j=1}w^2_j$

该正则项是为了降低模型的复杂度，第一项相当于加了个约束，当分裂后带来的提升小于这个惩罚性，则不继续分裂，第二项相当于对每个结点的得分做了个平滑操作。这两项的目的都是为了避免过拟合。

>  原式中的第三项，constant也是一种惩罚项，目的应该也是为了避免过拟合



### 与GBDT的区别

- XGBoost是GBDT的工程实现
- XGBoost显式的加入了正则项来控制模型的复杂度（在CART作为基分类器时）
- GBDT只涉及一阶导数的信息，XGBoost泰勒展开到了二阶导数
- XGBoost支持多种类型的基分类器（决策树、线性分类器等）
- GBDT迭代时用全部数据，XGBoost可以每次对数据做采样（样本采样、特征采样）
- GBDT没有特殊处理缺失值（必须人为补齐缺失值），XGBoost可以自动学习出缺失值的处理策略
- 不是区别：都用了shrinkage策略（具体做法就是衰减每个弱分类器的能力，给每个分类器前面加系数（0~1））
- 支持并行，训练前将每个特征的特征值进行排序并保存为block结构【在做划分点选择时，特征的增益计算可以并行进行】https://www.zhihu.com/question/280568070/answer/2553403899

> 处理缺失值：
>
> 1. 训练阶段
>    1. 分别计算把缺失值放到左子树和右子树的损失，选效果最好的
> 2. 测试阶段
>    1. 如果训练过程中，这个特征值出现过缺失的情况，就按训练过程中的划分方向
>    2. 如果训练集没出现过，那就默认左子树





### 注意点

- 不必做特征筛选。惩罚项有时候可以规避掉一些作用不大的特征。

- 不必对数据集做归一化。

  > 归一化是对连续特征来说的。那么连续特征的归一化,起到的主要作用是进行数值缩放。数值缩放的目的是解决梯度下降时,等高线是椭圆导致迭代次数增多的问题。而xgboost等树模型是不能进行梯度下降的,因为树模型是阶越的,不可导。树模型是通过寻找特征的最优分裂点来完成优化的。由于归一化不会改变分裂点的位置,因此xgboost不需要进行归一化。”



### 重要参数

```python
import xgboost as xgb
model = xgb.XGBClassifier()
# XGBClassifier, # binary:logistic， multi:softmax
# XGBModel,
# XGBRanker, # rank:pairwise
# XGBRegressor, # reg:squarederror
# XGBRFClassifier, # binary:logistic， multi:softmax
# XGBRFRegressor # reg:squarederror
```



n_estimators

训几棵树



booster: gbtree, gblinear, dart

迭代过程中运行的模型，一般用gbtree



learning_rate

学习率，一般设置0.05-0.3



min_child_weight

最小叶子结点数量，默认1



max_depth

树的最大深度，一般3-10



gamma

结点分割时需要降低多少loss，我理解就是那个公式里的constant



reg_alpha 与 reg_lambda

L1正则与L2正则项



objective

目标函数，可以客制（文档：https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html）



subsample

采样，控制采样出来的样本能占据到源数据集的多少数据量。数据集过少时，建议用1。



grow_policy

看代码中的文档描述，感觉有点像L1正则的系数，值越小越倾向于继续生长，那不就和reg_alpha冲突了吗。。没搞明白

>​    Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
>
>​    depth-wise. 1: favor splitting at nodes with highest loss change.



colsample_bytree

构建每棵树时的特征采样比例。





### 调参

- 利用xgboost的cv方法，可以在每一次迭代种交叉验证，返回理想的决策树数量。

- 对于给定决策树数量，进行决策树特定参数调优(max_depth, min_child_weight, gamma, subsample, colsample_bytree)。
- 调整正则化项的权重
- 降低学习率



### 别人的超参搜索过程

- 先找决策树的个数，先搜了个1~301，每次加10。发现在11的时候最好

![image-20230622194109749](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230622194109749.png)

- 再来个1~21，每次加1。发现8最好

![image-20230622194355038](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230622194355038.png)

- 调整max_depth，发现深度为8时最好

![image-20230625135153784](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230625135153784.png)

- 用交叉验证和网格搜索调整max_depth和min_child_weight【这里的分数我感觉是他交叉验证的分数，上面手动调参是测试集上（不合理，其实应该用验证集）的分数，反正他只是给个调参思路，咱们自己用的时候针对验证集调就行了】

![image-20230625135754649](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230625135754649.png)

- 上面搜出来的max_depth和自己手动调的差太多了，换个范围再搜一波

![image-20230625135926220](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230625135926220.png)

- 然后他固定了max_depth=8，min_child_weight=3【博主没解释为啥用这个组合，明明效果更差了。我猜可能是搜出了这个组合里的max_depth=8，那就用了这个组合？那为啥不直接固定max_depth=8，只针对min_child_weight搜索呢？】，下面调gamma

![image-20230625141018202](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230625141018202.png)

- 调subsample和colsample_bytree

![image-20230625141052387](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230625141052387.png)

- 调正则化参数reg_alpha和学习率learning_rate

![image-20230625141317031](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230625141317031.png)

- 博主说他这个全参数网格搜索的效果还不如上面手动调的。其实有大问题昂，他上面调的是在max_depth=8的时候，min_child_weight最好取3，他这个搜索范围根本就搜不到3，而且上面随机种子是27，现在改成25了，多少有点不严谨了

![image-20230625141812048](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230625141812048.png)



## LightGBM

视频：https://www.bilibili.com/video/BV13h4y147GK/?spm_id_from=333.337.search-card.all.click&vd_source=9f0a6e756906d2533e75947bbe7a2760

论文：https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

知乎： https://zhuanlan.zhihu.com/p/191677930?v_p=86

Github：https://github.com/microsoft/LightGBM

LightGBM是一个实现GBDT算法的框架，主要就是做了一堆优化策略来降低训练的开销（开销远低于XGBoost）



优化策略：连续变量离散化、互斥特征捆绑、基于梯度的单边采样

决策树建模优化方法：直方图算法（Histogram）、叶子节点优先的决策树生长策略（Leaf-wise tree growth）



### 基本原理



#### Histogram算法

- 先把连续的浮点**特征值离散化**成k个整数，构造一个宽度为k的直方图
- 遍历一遍数据，根据离散化的值作为索引，在直方图中累积统计量
- 根据直方图的离散值，遍历寻找最优的分割点

![image-20230626161205406](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230626161205406.png)

原本的排序算法，对每个特征值都要计算分裂的增益，直方图算法只需要计算k次，时间复杂度从O（#data * #feature) -> O(k * #feature)

其实该算法找到的分割点都不够精确，但最后结果都显示对最终精度影响不大



**直方图加速计算**

![image-20230626161742234](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230626161742234.png)





#### 叶子节点优先的决策树生长策略

**有深度限制**

选择具有最大误差的树叶进行生长，生长叶子的算法可以比基于层的算法减少更多loss。（Leaf-wise生长策略）

![image-20230626160253623](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230626160253623.png)

![image-20230626160203318](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230626160203318.png)



Leaf-wise容易导致过拟合，因此会增加一个最大深度限制来防止过拟合



#### 连续变量离散化（分箱）

根据特征取值确定上下限，根据设定的k值，等分成k个区间（超参）。例如：k=2, [0, 10] - > [0, 5)和[5, 10]，当x=2时，设定x=bin0；当x=6时，设定x=bin1。

![image-20230627105342407](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627105342407.png)

#### 互斥特征捆绑（降维）

**逆独热编码**

![image-20230627105706391](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627105706391.png)

**什么是互斥特征？**

不同为1（不同时为非0）

![image-20230627113607081](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627113607081.png)

**放宽互斥条件**

冲突比例（超参）

| $x_1$ | $x_2$ |
| ----- | ----- |
| 0     | 1     |
| 1     | 0     |
| 0     | 0     |
| 1     | 1     |

冲突比例 = 1 / 4 = 0.25

当冲突比例<设定的max_conflict_rate，就认为是可以捆绑的互斥特征





- **如何确定和哪个特征合并最好？**

Graph Coloring Problem（图着色问题）：给定一个无向图，如何用尽量少的颜色对图中的每个顶点进行着色，使得相邻顶点颜色不同。

**做法：**

把每个特征看成一个顶点，若某两个特征之间存在冲突，则形成一条边。

![image-20230627105342407](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627105342407.png)

根据x1_binned, x2_binned, x3, x4计算冲突比例

![image-20230627132943196](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627132943196.png)

构成的图

![image-20230627133147340](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627133147340.png)

根据设定的最大冲突比例，删除实际上会被认为是互斥的边

![image-20230627133448547](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627133448547.png)

根据degree从高到低的排序，先给x1着红色，然后给x2着绿色，然后给x3着黄色然后给x4着绿色（x2和x4冲突比例最低，让他俩相同颜色，也就是捆绑起来）



- **如何实现特征捆绑？**

> **论文原文：**假如有两个特征A和B，A的取值范围是[0, 10)而B的取值范围是[0, 20)，那就添加10个偏移量给B，使得B的取值范围是[10, 30)，这样A和B就不会冲突了，而且A和B合并以后的特征取值范围是[0, 30]。



x4要捆绑到x2上，x2的取值范围是[0, 1]，那最大值是1，那就把x4的取值范围（原来是[0, 1]）变成[1, 2]【实际上只对非0值操作】

```
那么x4的变化
[0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
->
[0, 0, 2, 0, 0, 2, 0, 2, 0, 2]

所以x2&x4变成
[1, 1, 0, 1, 0, 1, 1, 0, 0, 0] + [0, 0, 2, 0, 0, 2, 0, 2, 0, 2]
->
[1, 1, 2, 1, 0, 3, 1, 2, 0, 2]
```

![image-20230627113607081](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627113607081.png)

#### 基于梯度的单边采样

- 根据**梯度的绝对值**大小进行样本划分和抽样
- 实际抽取情况：梯度值绝对值最大前**a%**（top_rate）和对小梯度样本随机抽样**b%**（other_rate）
- a%越大，过拟合风险大，反之则欠拟合风险大，如果都很大则提高模型训练复杂度
- 实际上，后续决策树生长的过程中，小梯度样本的梯度会再乘一个大于1的**膨胀系数**，再和大梯度样本的梯度相加（用较小的数据量尽可能还原真实地数据集梯度，从而提升建模的精确度）
  - 膨胀系数$\frac{1-top\_rate}{other\_rate}$



### 并行优化

- 特征并行的主要思想是在不同机器在**不同的特征集合**上分别寻找**最优的分割点**，然后在机器间同步最优的分割点。

![image-20230627141942888](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627141942888.png)

- 数据并行则是让不同的机器先在**本地构造直方图**，然后进行全局的合并，最后在合并的直方图上面寻找最优分割点。
  - 合并局部直方图，得到全局直方图

![image-20230627142028642](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627142028642.png)

![image-20230627144051044](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627144051044.png)

- **优化**

  - 在特征并行算法中，**每个worker保存全部数据**，找到全局最佳切分点时，各个worker可以自行划分，不需要通信
  - 在数据并行中使用分散规约，把直方图合并的任务分摊到不同的机器，降低通信和计算，利用直方图做差，进一步减少了一半的通信录**【没看懂】**

  ![image-20230627143942339](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627143942339.png)

  - 基于投票的数据并行进一步优化数据并行中的通信代价，使通信代价变成常数级别，在数据量很大的时候，使用投票并行可以得到非常好的加速效果

![image-20230627144344182](http://dontquit.oss-cn-hangzhou.aliyuncs.com/myPic/image-20230627144344182.png)



### 与GBDT的区别

- 基于Histogram的决策树生成算法
- 大多数GBDT工具使用低效的按层生长的决策树生长策略，不加区分的对待同一层的叶子，有很多没必要的开销，很多叶子的分裂增益比较低，没必要进行搜索和分裂。LightGBM使用了带有深度限制的按叶子生成的算法。
- 单边梯度采样（GOSS），减少大量具有小梯度的数据实例，在计算信息增益的时候只利用剩下的具有高梯度的数据，相比XGBoost节省了不少开销
- 互斥特征捆绑，可以将一些互斥的特征绑定为一个特征，从而实现降维

- 直接支持类别特征，不需要转成one-hot向量（直方图算法，分桶操作带来的优势）
- 支持高效并行
- Cache命中率优化



### 代码

上面提到的一些重要超参

- **剪枝类**

分桶个数k：max_bin

num_leaves：叶子节点数量

max_depth：深度限制

min_split_gain：分裂所需的增益

min_child_weight：子结点最小权重和，权重太小就没必要生长了

min_child_samples：单个叶子结点上的最小样本数量

- **过程控制类**

n_estimators：基学习器数量

learning_rate：学习率

top_rate：sklearn没有，使用goss算法时会自动调整

other_rate: sklearn没有，使用goss算法时会自动调整

subsample_for_bin: 分桶时抽样的样本个数（如果用goss就不看这个参数）

正则参数：reg_alpha, reg_lambda

boosting_type: gbdt, dart, goss, rf

gbdt比较稳定

dart适合噪声比较多的数据集

goss适合很大很复杂的数据集

- **特征与数据处理类**

subsample：抽样的样本比例

subsample_freq：抽样频率，每隔几轮抽样一次

colsample_bytree：抽样的特征比例

- **其它**

objective, importance_type (split，使用的特征在模型中被选中的作为分裂特征的次数；gain，使用增益)，n_jobs（并行线程数）



```python
#导入所需要的包
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report#评估报告
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
import matplotlib.pyplot as plt#可视化
import seaborn as sns#绘图包
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler#归一化，标准化

from sklearn.metrics import precision_score
import lightgbm as lgb  
```



```python
df=pd.read_csv(r"数据.csv")
 
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)
model=lgb.LGBMClassifier(n_estimators=39,max_depth=8,num_leaves=12,max_bin=7,min_data_in_leaf=10,bagging_fraction=0.5,
                             feature_fraction=0.59,boosting_type="gbdt",application="binary",min_split_gain=0.15,
                             n_jobs=-1,bagging_freq=30,lambda_l1=1e-05,lambda_l2=1e-05,learning_rate=0.1,
                         random_state=90)
model.fit(X_train,y_train)
# 预测值
y_pred = model.predict(X_test)
'''
评估指标
'''
# # 求出预测和真实一样的数目
true = np.sum(y_pred == y_test )
print('预测对的结果数目为：', true)
print('预测错的的结果数目为：', y_test.shape[0]-true)
# 评估指标
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
print('预测数据的准确率为： {:.4}%'.format(accuracy_score(y_test,y_pred)*100))
print('预测数据的精确率为：{:.4}%'.format(
      precision_score(y_test,y_pred)*100))
print('预测数据的召回率为：{:.4}%'.format(
      recall_score(y_test,y_pred)*100))
# print("训练数据的F1值为：", f1score_train)
print('预测数据的F1值为：',
      f1_score(y_test,y_pred))
print('预测数据的Cohen’s Kappa系数为：',
      cohen_kappa_score(y_test,y_pred))
# 打印分类报告
print('预测数据的分类报告为：','\n',
      classification_report(y_test,y_pred))
 
# ROC曲线、AUC
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
# 预测正例的概率
y_pred_prob=model.predict_proba(X_test)[:,1]
# y_pred_prob ,返回两列，第一列代表类别0,第二列代表类别1的概率
#https://blog.csdn.net/dream6104/article/details/89218239
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob, pos_label=2)
#pos_label，代表真阳性标签，就是说是分类里面的好的标签，这个要看你的特征目标标签是0,1，还是1,2
roc_auc = metrics.auc(fpr, tpr)  #auc为Roc曲线下的面积
# print(roc_auc)
plt.figure(figsize=(8,6))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.plot(fpr, tpr, 'r',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.xlabel('False Positive Rate') #横坐标是fpr
plt.ylabel('True Positive Rate')  #纵坐标是tpr
plt.title('Receiver operating characteristic example')
plt.show()
```



调参过程参考XGBoost就好了，用网格搜索&学习曲率找到最优点

