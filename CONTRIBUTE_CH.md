# 贡献规范

[English version](CONTRIBUTE.md)

## 导引

首先感谢您关注 EduKTM 并致力于让其变得更好！
在您开始贡献自己的一份力之前，需要注意以下几点：

1. 如果您希望我们实现新的功能。
   - 可以在通过 issue 来告诉我们您想要的功能，我们将及时展开讨论设计和实现。
   - 一旦我们一致地认为这个计划不错，那么您可以期待新的功能很快就可以与您见面。
2. 如果您想要对于某个未解决问题的 issue 提供解决性意见或 bug 修复。
   - 可以先在 [EduKTM issue list](https://github.com/bigdata-ustc/EduKTM/issues) 中搜索您的问题。
   - 之后，选择一个具体问题和评论，来提供您的解决性意见或者 bug 修复。
   - 如果对于具体的 issue，您需要更多的细节，请向我们咨询。

一旦您实现并已经测试过了你的想法或者是对于 bug 的修复，请通过 Pull Request 提及到到 [EduKTM](https://github.com/bigdata-ustc/EduKTM) :
1. 首先fork此仓库到你的分支下
2. 对代码进行修改。注意：我们强烈建议你遵守我们的 [commit格式规范](CONTRIBUTE_CH.md#关于Commit的格式)
3. 通过代码测试，测试覆盖度达到100%，例子可见[此处](tests/dkt)
4. 通过Pull Request 提及到到 [EduKTM](https://github.com/bigdata-ustc/EduKTM) 。注意：我们提供了一个标准的PR请求模板，你需要认真完成其中的信息，一个标准且规范的PR可参考[此处]()

以下是对于不同贡献内容的有用建议：

### 添加新的数据集或者数据分析

有关新数据集或数据分析，请移步至 [EduData](https://github.com/bigdata-ustc/EduData) 。

### 添加新的 KTM 模型

新实现的 KTM 模型需要：
1. 数据集的预处理。
2. 继承 `EduKTM/meta.py` 中的的 `class KTM`，并实现中间的四个方法。
3. 编写模型对应的 example 代码（这里指的是可供其他人运行测试使用的 demo），例子可见[此处](examples/DKT)：至少应当包括：[notebook](examples/DKT/DKT.ipynb) 和 [script](examples/DKT/DKT.py)
4. 编写模型对应的测试代码，保证测试覆盖度为100%，例子可见[此处](tests/dkt)

#### 数据预处理

关于数据集的预处理，我们提供如下两种建议：

1. 编写一个 script，完成：
   - 对原始数据集中进行处理，转换。
   - 训练/验证/测试集划分。
2. 提交或使用 [KTBD](https://github.com/bigdata-ustc/EduData) 数据集（已划分好训练/验证/测试集）。

#### 模块编写

编写的新 KTM 模型，其中几个重要模块需要继承 `EduKTM/meta.py` 中的 `class KTM`。
需要注意的是，我们并不对您的神经网络、算法（例如，网络构造、优化器、损失函数定义等）进行约束。

- 训练模块

该模块为训练模块，用于对模型、算法进行训练。

```python3
    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

- 测试模块

该模块为测试模块，用于对模型、算法进行验证、测试。

```python3
    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

- 模型存储模块

该模块为存储模块，用于保存训练好了的模型、算法。

```python3
    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

- 模型读取模块

该模块为模型读取模块，用于读取保存好了的模型、算法。

```python3
    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

#### 编写 Demo

编写模型对应的 Example 代码，例子可见[此处](examples/DKT) ：

#### 代码注释风格

请使用 Numpy 代码注释风格：

```
function 的功能

    Parameters
    ----------
    变量名 1: 类型, 是否 optional
       描述
    变量名 2: 类型, 是否 optional
       描述
    ...

    Returns
    -------
    变量名: 类型
       描述

    See Also (可选)
    --------
    类似 function: 类似 function 的功能

    Examples (可选)
    --------
    >>> 举例怎么用
```

### 关于Commit的格式

#### commit format

```
[<type>](<scope>) <subject>
```

#### type
- `feat`：新功能（feature）。
- `fix/to`：修复 bug，可以是 Q&A  发现的 bug，也可以是自己在使用时发现的 bug。
   - `fix`：产生 diff 并自动修复此问题。**适合于一次提交直接修复问题**。
   - `to`：只产生 diff 不自动修复此问题。**适合于多次提交**。最终修复问题提交时使用 `fix`。
- `docs`：文档（documentation）。
- `style`：格式（不影响代码运行的变动）。
- `refactor`：重构（即非新增功能，也不是修改 bug 的代码变动）。
- `perf`：优化相关，比如提升性能、体验。
- `test`：增加测试。
- `chore`：构建过程或辅助工具的变动。
- `revert`：回滚到上一个版本。
- `merge`：代码合并。
- `sync`：同步主线或分支的 bug。
- `arch`: 工程文件或工具的改动。

#### scope (可选)

scope 是用于说明 commit 影响的范围，比如<u>数据层</u>、<u>控制层</u>、<u>视图层</u>等等，视项目不同而不同。

例如在 Angular，可以是 location，browser，compile，compile，rootScope， ngHref，ngClick，ngView等。如果你的修改影响了不止一个scope，你可以使用`*`代替。

#### subject (必须)

subject 是 commit 目的的简短描述，不超过50个字符。

结尾不加句号或其他标点符号。

#### Example

- **[docs] update the README.md**

```sh
git commit -m "[docs] update the README.md"
```

## FAQ

问题: 我已经在本地仔细地测试了代码，并通过了代码检查，但是在 CI 步骤时却报错？
回答: 这个问题可能是两个原因造成： 
1. 在线的 CI 系统与您自己本地系统有差别；
2. 可能是网络原因造成的，如果是可以通过 CI 的日志文件查看。