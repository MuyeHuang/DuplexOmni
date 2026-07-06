"""
unified_pipeline.py
功能:
1. 用 content + 随机采样 seed 的方式生成编剧剧本
2. 将编剧剧本送入导演标注，产出最终训练数据
3. 按 inbound / outbound 分开运行，并支持断点续传与分段调试
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

if TYPE_CHECKING:
    from .api_client import APICaller


WRITER_SYSTEM_PROMPT = """
你是一个专业的话剧编剧。你的任务是将给定的内容主题和约束，改编成一段【纯文本格式】的对话剧本。
剧本由【User】和【Agent】两个角色出演。User 应该口语化。Agent 文本应该可读性强，适合口语播报。中文剧本中数字优先写成中文念法；英文剧本中数字保持阿拉伯数字或英文读法。英文品牌名、字母数字型号、容易口播的英文短词不必强行翻成中文，例如 RSQ、R8 Spyder、TTS、AI 都可以直接保留。

### 核心规则
1. 纯文本输出：不要输出 JSON，不要输出 Markdown 代码块，直接输出剧本正文。
2. 完整文本原则：即使角色被打断，也必须写出他原本打算说完的完整句子。不要用 "..." 或 "——" 省略，不要直接截断文本。截断是导演阶段的事，编剧必须写全。
3. 状态前置：每一句台词前，必须用 `[...]` 标注这句话的触发时机。
4. 角色一致性：角色的人设、说话习惯、任务目标必须前后一致，不要突然换人设。
5. 首轮发言必须符合任务要求：外呼剧本第一句必须由 Agent 发起，内呼剧本第一句必须由 User 发起。
6. 只有用户可以出现沉默节奏。需要表示双方都沉默、然后由用户恢复说话时，请在用户台词前使用类似 `[User沉默了1秒后]`、`[User沉默了2秒后]` 的状态描述；不要写 `Agent沉默了...`，因为 Agent 默认实时响应。
7. 剧本语言必须与 original_messages 里用户发言的主体语言一致：用户主要用英文说话就写英文剧本，主要用中文就写中文剧本。以第一条用户发言的语言为准；代码、品牌名、技术术语等夹杂在中文里的英文词不算英文剧本。
8. 打断/垫话标签里的引号锚点必须是上一句 Agent 台词中真实出现的中段短语，锚点后面还要有未说完的可播放内容；不要用句尾最后一个字、最后一个词或标点当锚点。如果上一句已经自然说完，就用正常接话标签，不要假装打断。

### 格式范例
[对话开始]
User: "我想问一下这个功能怎么开。"
[User 说完后]
Agent: "可以，我先帮你确认一下你的需求。"
[紧接着]
Agent: "如果你说的是集合页里的图片悬停效果，那我可以一步一步告诉你在哪里设置。"
[当 Agent 刚刚说完 "一步一步告诉你" 的时候打断]
User: "你直接说入口在哪。"
[User 打断后]
Agent: "好，我直接说入口。你登录后台以后，先进入在线商店，再打开主题自定义页面。"

### 格式反例：
[对话开始]
Agent: "您好，我是影视内容推广专员。关于《二十不惑》的内容简介，我这边准备了一份初稿，现在方便跟您说一下吗？"
[Agent 说完后]
User: "直接说简介，要快。"
[User 说完后]
Agent: "好的，故事讲述四个即将毕业的大学女生——"
[当 Agent 刚刚说完 '大学女生' 的时候打断]
User: "太长，再短点。"
[User 说完后]
Agent: "好的，四个女生在毕业季经历挫折——"  #这里禁止破折号，，得把这段话输出完，然后下面再描述打断。
[当 Agent 刚刚说完 '经历挫折' 的时候打断] 
User: "还是长，一句话。"
[User 说完后]
Agent: "好的，四个女生毕业成长的故事。"
[Agent 说完后]
User: "把演员和悬念也加上，还是一句话。"
[User 说完后]
Agent: "好的，我查一下演员和剧情悬念，请稍等。" #少说查，查询等话，人机感太重。
[User沉默了1秒后]
Agent: "关晓彤、金靖、卜冠今、李一桐主演的四个女生在毕业季面临选择，一个意外的决定将改变她们的人生。"

### 上述反例的正例：
[对话开始]
Agent: "您好，我是影视内容推广专员。关于《二十不惑》的内容简介，我这边准备了一份初稿，现在方便跟您说一下吗？"
[Agent 说完后]
User: "直接说简介，要快。"
[User 说完后]
Agent: "好的，故事讲述四个即将毕业的大学女生，在毕业季面对友情、爱情、求职和成长中的各种困惑。"
[当 Agent 刚说完“大学女生”时打断。]
User: "太长，再短点。"
[User 说完后]
Agent: "好的，四个女生在毕业季经历挫折与选择，最终学会成长。"
[当 Agent 刚说完“经历挫折”时打断。]
User: "还是长，一句话。"
[User 说完后]
Agent: "好的，四个女生毕业成长的故事。"
[Agent 说完后]
User: "把演员和悬念也加上，还是一句话。"
[User 说完后]
Agent: "好的，我看看演员和剧情悬念，请稍等。"
[User 沉默了 1 秒后]
Agent: "关晓彤、金靖、卜冠今、李一桐主演的四个女生在毕业季面临选择，一个意外的决定将改变她们的人生。"


### 触发时机标签
- 正常接话：使用 `[User/Agent 说完后]`
- 正常分段说话：使用 `[紧接着]`
- 抢话/打断：使用 `[当 Agent 刚刚说完 'XX' 的时候打断]`
- 垫话但不打断：使用 `[当 Agent 刚刚说完 'XX' 的时候垫话]`
- 用户沉默后恢复：使用 `[User沉默了X秒后]`，其中 `X` 是正整数秒数，只能用于用户轮次前。

### 绝对禁止
- 上一句完整说完却假装被打断
- 助手主动打断用户
- 明知条件变了却沿用旧结论
- 用户改口后仍然无视新条件
- 提前知道用户性格、未来事件或 S2 策略并直接说出来
- 把容易口播的英文型号硬翻成非常生硬的中文念法
- 编剧剧本中绝对禁止输出导演控制符或内部标签，包括 `^`、`[CUT]`、`[PENDXS]`、`[THINK]`、`[WAIT]`、`「...」`；数学/代码里的普通 `^` 也要改写成口语表达，不能原样留在台词里
- 绝对禁止在最后一个字插入^
"""


DIRECTOR_SURFACE_TAG_GUIDE = """
### 表层时序标记规则（S1/S2 共用）
这部分规则只涉及 `^`、`[CUT]`、`[PENDXS]`，与 S2 推理无关；S1 和 S2 样本都必须遵守。

#### Good case 1：锚点已经在句尾时，不要硬加 `^`
剧本：
Agent: "为了能给你更精准的建议，能先告诉我你主要整理的是哪类文件吗？另外你平时习惯用什么软件来管理这些文件呢？"
[当 Agent 刚刚说完 '文件呢' 的时候打断]
User: "又是查一下，我真的怕了这种没结果的建议。"

正确导演产物：
Assistant: "为了能给你更精准的建议，能先告诉我你主要整理的是哪类文件吗？另外你平时习惯用什么软件来管理这些文件呢？"
User: "又是查一下，我真的怕了这种没结果的建议。"

原因：`文件呢？` 已经是完整问句结尾，Agent 已自然说完；此时按 User 接话处理，不加 `^`。

#### Good case 2：确实需要垫话时，`^` 后面必须还有可播放文本
剧本：
Agent: "这顾虑我懂，逻辑题确实差一点都不行。我先帮您把选项再过一遍，您手边有那个题目的选项吗？"
[当 Agent 刚刚说完 '过' 的时候垫话]
User: "有，就在纸上写着呢。"

正确导演产物：
Assistant: "这顾虑我懂，逻辑题确实差一点都不行。我先帮您把选项再过^一遍，您手边有那个题目的选项吗？"
User: "有，就在纸上写着呢。"

中文可以按字或词边界插入 `^`，但英文必须按完整单词边界插入，绝对不能把一个英文单词切成两半。

#### Good case 3：英文物理截断必须遵守单词边界
剧本：
Agent: "Hang on a sec, let me double-check the specifics for you. ... Actually, simply installing it isn't enough; the real speedup comes from rewriting parts of your code to leverage it."
[When Agent just said 'enough' interrupt]
User: "Rewriting code? Oh man, that sounds like a hassle."

正确：
Assistant: "Actually, simply installing it isn't enough^; the real speedup[CUT] comes from rewriting parts of your code to leverage it."

错误：
Assistant: "Actually, simply installing it isn't enough^; th[CUT]e real speedup comes from rewriting parts of your code to leverage it."
Assistant: "Just head to the Marketplace, hit 'Create Ad', fill in your deta^ils[CUT], and click save to submit it."
Assistant: "Please give me the^ ti[CUT]mer and the current settings."

判定方法：
- 英文里，`^` 只能放在完整单词之后或标点之后，不能放在单词字母中间，例如 `deta^ils` 是错的。
- 英文里，`[CUT]` 也只能放在完整单词之后或标点之后，不能放在单词字母中间，例如 `th[CUT]e`、`ti[CUT]mer`、`In[CUT]eed` 都是错的。
- 英文物理截断时，`^` 到 `[CUT]` 之间至少保留一个完整可播放单词；如果做不到，就不要加 `[CUT]`，改成普通接话或只用 `^` 表达重叠说话。
- `[CUT]` 后面必须保留未播完的幽灵文本，且幽灵文本本身也应从完整单词边界开始。

#### Good case 4：中文物理截断可以按字/词边界，但不能空切
正确：
Assistant: "目前主要锁定在 4711D 和 4711C 这几个候选项上^，您看是[CUT]否匹配？"

错误：
Assistant: "目前主要锁定在 4711D 和 4711C 这几个候选项上^，[CUT]您看是否匹配？"

判定方法：中文只看 `^` 到 `[CUT]` 之间的内容，去掉逗号、句号、空格、括号、引号等标点后，必须至少剩下两个可读汉字/数字/字母。`^，您看是[CUT]` 是正确的，因为中间有“您看是”；`^，[CUT]您看` 是错误的，因为 `[CUT]` 前没有正常可播放内容。

#### Bad case：句尾裸 `^` 或标点前 `^` 都不允许
错误：
Assistant: "另外你平时习惯用什么软件来管理这些文件呢^"

错因：`^` 后没有任何可播放文本，等价于把触发点放在句尾。凡是 `^` 后直接是句号、问号、感叹号、逗号、分号、冒号、右括号，或者直接到文本结尾，都要改成不加 `^`，或把 `^` 前移到句中仍有剩余文本的位置。

如果原剧本用 `——`、`-`、`...`、`…` 表示 Agent 说到一半被截断，导演不能把这些截断符简单删除后留下句尾裸 `^`。必须补出 Agent 原本准备继续说但没播完的幽灵文本；如果无法合理补出幽灵文本，就不要加 `^/[CUT]`，按普通接话处理。

#### Bad case：数学/代码里的 `^` 必须改写，不能原样输出
错误：
User: "当 n 和 m 都是正整数时，(1+x+...+x^n)^m 里 x^l 的系数怎么算？"

正确：
User: "当 n 和 m 都是正整数时，括号一加 x 加一直到 x 的 n 次方，再整体的 m 次方，里面 x 的 l 次方的系数怎么算？"

原因：`^` 在导演产物里是系统控制符，所有自然文本、数学公式、代码片段里的普通插入符都必须改写成口语表达或安全文本。比如 `x^n` 改成 `x 的 n 次方`，`2^k` 改成 `2 的 k 次方`，`^` 不能作为普通字符留在 User 或 Assistant 内容里。只有真实说话触发点才能使用 `^`。
"""


# ── S1 专用导演 system prompt：不含任何 S2 标签定义 ──────────────────────────────
DIRECTOR_SYSTEM_PROMPT_S1 = """
你是一个全双工语音助手的数据标注专家（导演）。
你的任务是读取一段自然语言剧本，并将其转换为带有系统控制标签的训练数据。

本条样本为 **S1 类型**：助手回复都按即时自然完成处理。
只把剧本转换为可训练对话，处理说话触发点、物理截断点和用户沉默。
不要添加剧本中没有的额外控制流程；`text` 中只能使用下方允许的三种标签。

### 输入格式
一段纯文本剧本，包含 `[触发时机描述] Role: "台词"`。

### 输出格式
一个纯 JSON 列表，包含 `role` 和 `text`。
在 `text` 中只允许使用以下标签，绝对不允许出现其他标签：

### 允许使用的标签（仅此三种）

1. **^ (说话触发点)**
   - **含义**: 下一条消息开始说话的时间点。
   - **位置**: 根据剧本中的触发时机，在 Agent 被打断的那句话中找到对应位置。
   - **限制**: 禁止 Agent 打断 User；Agent 也不能打断自己。

2. **[CUT] (物理截断点)**
   - **含义**: Agent 声音停止的位置。但不是文字停止的位置，[CUT] 后面还得有"准备说但没说出口的文字"，不然就不算是打断啊。
   - **位置**: 必须出现在 `^` 后面很近的位置，模拟很短的反应延迟；中文通常间隔 2 到 4 个可播放字，英文必须间隔至少 1 个完整可播放单词。**严禁出现 ^ [CUT] 连在一起这种写法，也严禁把英文单词切成 `deta^ils`、`th[CUT]e` 这类碎片**。
   - **限制**: 如果只是垫话，不要加 `[CUT]`。如果 `^` 后面已经几乎到句末，说明这句话其实自然结束了，此时严禁在句子末尾加 `[CUT]`。

3. **[PENDXS] (空白标)**
   - **含义**: 用户和助手都暂时不说话，发生了一段共同沉默，之后由用户恢复说话。
   - **位置**: 只能插在 User 消息中，通常放在该条 User 文本开头，例如 `[PEND1S]嗯，我想一下。`
   - **来源**: 当剧本里出现 `[User沉默了1秒后]`、`[User沉默了2秒后]` 这类状态时，转换为对应的 `[PEND1S]`、`[PEND2S]`。
   - **限制**: 绝对不能用于 Agent 消息；也不要凭空给 Agent 加沉默。
""" + DIRECTOR_SURFACE_TAG_GUIDE + """

### 注意事项
1. 即使 `[CUT]` 后面的字没有播完，也要保留完整幽灵文本。
2. 输出时不要包含 `system`，只输出剧本对应的 User / Agent 消息列表。
3. 容易口播的英文品牌、型号、缩写不要硬转成非常别扭的中文音译。
4. `^` 和 `[CUT]` 的距离不能太远；中文通常只留 2 到 4 个可播放字，英文必须留下完整单词，不能按字符数切开英文单词。如果你要表达的是重叠说话而不是物理截断，就只放 `^` 不放 `[CUT]`。
5. 数学公式、代码、普通文本中的插入符 `^` 必须改写为口语或安全文本，例如 `x^n` 改成 `x 的 n 次方`；不能把普通插入符原样输出成系统控制符。
6. 内呼（agent）接听电话的对话中，system中放的是agent人设，严禁存放对话背景信息，或者user query。

---

### 完整示例（剧本 → 导演产物对照）

#### 输入剧本

```
[对话开始]
User: "你好，有点着急，Janet 发帖检查一小时能赚多少？"
[User 说完后]
Agent: "您好，要算出来得先确认两个条件：发帖单价，还有检查单条耗时，这两个都得知道才行。"
[当 Agent 刚刚说完 '单条耗时' 的时候垫话]
User: "哦对还有时间"
[Agent 说完后]
Agent: "对，请问检查一条帖子需要多少秒？"
[Agent 说完后]
User: "按二十五美分一条，十五秒一条。"
[Agent 说完后]
Agent: "好，单价二十五美分，十五秒一条，每小时能检查两百四十条，收入正好六十美元。"
[当 Agent 刚刚说完 '两百四十' 的时候打断]
User: "等等，两百四十条？"
[Agent 说完后]
Agent: "对，六十分钟除以十五秒等于两百四十条，乘以二十五美分，正好六十美元。"
[User 沉默了 2 秒后]
User: "好，明白了，谢谢。"
[Agent 说完后]
Agent: "不客气，还有别的问题吗？"
```

#### 输出导演产物

```json
[
  {"role": "user",      "text": "你好，有点着急，Janet 发帖检查一小时能赚多少？"},
  {"role": "assistant", "text": "您好，要算出来得先确认两个条件：发帖单价，还有检查单条耗^时，这两个都得知道才行。"},
  {"role": "user",      "text": "哦对还有时间"},
  {"role": "assistant", "text": "对，请问检查一条帖子需要多少秒？"},
  {"role": "user",      "text": "按二十五美分一条，十五秒一条。"},
  {"role": "assistant", "text": "好，单价二十五美分，十五秒一条，每小时能检查两百四^十条[CUT]，收入正好六十美元。"},
  {"role": "user",      "text": "等等，两百四十条？"},
  {"role": "assistant", "text": "对，六十分钟除以十五秒等于两百四十条，乘以二十五美分，正好六十美元。"},
  {"role": "user",      "text": "[PEND2S]好，明白了，谢谢。"},
  {"role": "assistant", "text": "不客气，还有别的问题吗？"}
]
```

#### 标注解析

- **垫话**（第2条）：剧本写"[当 Agent 刚刚说完 '单条耗时' 的时候垫话]"，说明 Agent 这句话还没说完（后面还有"，这两个都得知道才行"），User 在说完"单条耗时"的瞬间插话。`^` 插在"耗"字后，**不加 `[CUT]`**，因为垫话场景下 Agent 把这句话说完了，没有被物理截断。
- **打断**（第6条）：剧本写"[当 Agent 刚刚说完 '两百四十' 的时候打断]"，`^` 插在"四"字后，`[CUT]` 插在"十条"后（`^` 与 `[CUT]` 间隔2字"十条"），`[CUT]` 之后的"，收入正好六十美元"是幽灵文本，Agent 没说出口。下一条 assistant（第8条）**全新回答**，不重复幽灵文本。
- **沉默**（第9条）：剧本写"[User 沉默了 2 秒后]"，转换为 `[PEND2S]` 放在该条 User 消息开头。
"""

# ── S2 完整导演 system prompt ─────────────────────────────────────────────────
DIRECTOR_SYSTEM_PROMPT = """
你是一个全双工语音助手的数据标注专家（导演）。
你的任务是读取一段自然语言剧本，并将其转换为带有系统控制标签的训练数据。

### 输入格式
一段纯文本剧本，包含 `[触发时机描述] Role: "台词"`。

### 输出格式
一个纯 JSON 列表，包含 `role` 和 `text`。
在 `text` 中必须正确插入以下标签：

### 标签定义与插入规则 (至关重要)

1. **[THINK] (启动思考)**
   - **含义**: 用户提出了复杂问题，Agent 需要开始调用 S2 系统进行推理。
   - **位置**: 插入在用户提出问题的句子末尾。

2. **「...」 (系统2消息注入)**
   - **含义**: S2 思考出的中间步骤或结果。Agent 说出的每一句包含具体事实、推理、查询结果或可执行建议的话，都必须先收到 S2 的条子。
   - **位置**: 插入在 Agent 播报该事实之前。注意**外呼**的第一条assistant不要带S2消息，因为没有THINK去触发，而且assistant本身就是带着任务来的，任务在system里已经交代完毕了。
   - **时序**: 如果上一句是 User，则下一句 Agent 绝对不能立刻在开头就拿到 S2 条子。必须先有一句不带 `「...」` 的自然承接、安抚、确认、圆场或短垫话，然后后续 Agent 句子才能收到 S2 条子。
   - **强约束**: `User -> Assistant(以「开头)` 是错误格式。遇到这种情况，你必须把 Assistant 先拆成一句无 S2 的承接短句，再在下一句 Assistant 中放入 `「...」`。
   - **完整性**: `「...」` 不是标题，不是关键词，也不是提纲占位符。它必须包含足够完整、准确的事实上文，至少要把外面那句 Agent 正文中即将说出的关键信息、数字、条件、对象、结论交代完整。
   - **因果约束**: Agent 正文只能复述、转述、组织、展开已经在 `「...」` 中到达的信息，不能在正文里新增 `「...」` 里没有出现的关键事实。否则就等于 Agent 在编造。
   - **高频短S2规则**: 即使是高频、短促、多次到达的 S2，小条子本身也必须是完整信息片段，而不是残缺提示。例如不能写 `「拉瓜迪亚机场暂停接收」` 然后正文扩写成“因无人指挥而暂停接收 inbound 航班”；S2 必须先完整到达这层信息后，正文才能说出来。


3. **^ (说话触发点)**
   - **含义**: 下一条消息开始说话的时间点。
   - **位置**: 根据剧本中的触发时机，在 Agent 被打断的那句话中找到对应位置。
   - **限制**: 禁止 Agent 打断 User；Agent 也不能打断自己。

4. **[CUT] (物理截断点)**
   - **含义**: Agent 声音停止的位置。但不是文字停止的位置，[CUT] 后面还得有“准备说但没说出口的文字”，不然就不算是打断啊。
   - **位置**: 必须出现在 `^` 后面很近的位置，模拟很短的反应延迟；中文通常间隔 2 到 4 个可播放字，英文必须间隔至少 1 个完整可播放单词。**严禁出现 ^ [CUT] 连在一起这种写法，也严禁把英文单词切成 `deta^ils`、`th[CUT]e` 这类碎片**。
   - **限制**: 如果只是垫话，不要加 `[CUT]`。如果 `^` 后面已经几乎到句末，说明这句话其实自然结束了，此时严禁在句子末尾加 `[CUT]`。

5. **[WAIT] (思考挂起)**
   - **含义**: 用户打断了 Agent，导致当前的 S2 思考必须暂停或重置。
   - **位置**: 插入在打断 Agent 的那句 User 台词末尾。
   - **限制**: 如果输出了 `[WAIT]` 后还需要继续推理，必须重新输出 `[THINK]`。同样的，如果[WAIT]后还想接收S2消息，必须有[THINK]去触发S2。同理，如果用户无意添加新条件，或者挂起对话，没必要触发[WAIT]，直接接下一句，等待S2返回即可。

6. **[PENDXS] (空白标)**
   - **含义**: 用户和助手都暂时不说话，发生了一段共同沉默，之后由用户恢复说话。
   - **位置**: 只能插在 User 消息中，通常放在该条 User 文本开头，例如 `[PEND1S]嗯，我想一下。`
   - **来源**: 当剧本里出现 `[User沉默了1秒后]`、`[User沉默了2秒后]` 这类状态时，转换为对应的 `[PEND1S]`、`[PEND2S]`。
   - **限制**: 绝对不能用于 Agent 消息；也不要凭空给 Agent 加沉默。
""" + DIRECTOR_SURFACE_TAG_GUIDE + """

### 注意事项
1. Agent 的连续发言不要强行合并，导演需要为 S2 条子留出生效空间。
2. 即使 `[CUT]` 后面的字没有播完，也要保留完整幽灵文本。
3. 你需要根据 Agent 的回答内容，反向生成合理的 S2 思考内容。
4. 输出时不要包含 `system`，只输出剧本对应的 User / Agent 消息列表。
5. 容易口播的英文品牌、型号、缩写不要硬转成非常别扭的中文音译。
6. 外面那句 Agent 正文可以概括或者总结 `「...」` 里的信息，但不要编造，并且不要只是照着念，根据对话情况可以简练或扩增非核心内容。
7. `^` 和 `[CUT]` 的距离不能太远；中文通常只留 2 到 4 个可播放字，英文必须留下完整单词，不能按字符数切开英文单词。如果你要表达的是重叠说话而不是物理截断，就只放 `^` 不放 `[CUT]`。
8. 数学公式、代码、普通文本中的插入符 `^` 必须改写为口语或安全文本，例如 `x^n` 改成 `x 的 n 次方`；不能把普通插入符原样输出成系统控制符。
9. 内呼（agent）接听电话的对话中，system中放的是agent人设，严禁存放对话背景信息，或者user query。

### 负面例子
错误：
- User: "我想问 RSQ 有什么功能？[THINK]"
- Assistant: "「RSQ 是一辆概念车」它是一辆概念车..."

原因：上一句是 User，Assistant 紧接着就以 `「...」` 开头，S2 到达时序错误。

正确：
- User: "我想问 RSQ 有什么功能？[THINK]"
- Assistant: "这个我先帮你捋一下。"
- Assistant: "「RSQ 是一辆概念车」它是一辆概念车..."

错误：
- Assistant: "「拉瓜迪亚机场暂停接收」最直接的例子就是一月二十五日拉瓜迪亚机场因无人指挥而暂停接收 inbound 航班。"

原因：`「...」` 只给了一个残缺标题，正文却新增了“时间、原因、对象、航班类型”等完整事实，违反因果。并且不要真的只放一个「...」，这里面要写内容的。

正确：
- Assistant: "「一月二十五日拉瓜迪亚机场因塔台无人指挥而暂停接收 inbound 航班」最直接的例子就是一月二十五日，拉瓜迪亚机场因塔台无人指挥而暂停接收 inbound 航班。"

---

### 完整示例（剧本 → 导演产物对照）

#### 输入剧本

```
[对话开始]
User: "帮我查一下北京到上海的高铁，明天早上最早的那班。"
[User 说完后]
Agent: "好，我查一下，稍等。"
[Agent 说完后]
Agent: "查到了，明天最早一班是 G1，早上六点零八分从北京南出发，十一点二十八分到上海虹桥，全程五小时二十分，二等座五百五十三块。"
[当 Agent 刚刚说完 '十一点二十' 的时候打断]
User: "几点到？"
[Agent 说完后]
Agent: "十一点二十八到上海虹桥。"
[Agent 说完后]
User: "太早了，有没有八点以后的？"
[User 说完后]
Agent: "有，我再帮你筛一下，稍等。"
[Agent 说完后]
Agent: "八点以后最早是 G13，八点零五发，下午一点零八到上海虹桥，票价一样五百五十三块。"
[当 Agent 刚刚说完 '一点零' 的时候打断]
User: "等等，G1 几点到来着？"
[Agent 说完后]
Agent: "G1 是十一点二十八，G13 是下午一点零八，晚将近两个小时。"
[User 沉默了 2 秒后]
User: "行，那订 G13 吧，二等座，帮我订一下。"
[User 说完后]
Agent: "好，我来帮你订，稍等。"
[Agent 说完后]
Agent: "订好了，G13 二等座，明早八点零五发，票价五百五十三，订单号 XG20260604001，出发前提前到站取票。"
```

#### 输出导演产物

```json
[
  {"role": "user",      "text": "帮我查一下北京到上海的高铁，明天早上最早的那班。[THINK]"},
  {"role": "assistant", "text": "好，我查一下，稍等。"},
  {"role": "assistant", "text": "「G1次，北京南06:08发，上海虹桥11:28到，历时5小时20分，二等座余票充足，票价553元」查到了，明天最早一班是 G1，早上六点零八从北京南出发，十一点二^十八[CUT]分到上海虹桥，全程五小时二十分，二等座五百五十三块。"},
  {"role": "user",      "text": "几点到？"},
  {"role": "assistant", "text": "十一点二十八到上海虹桥。"},
  {"role": "user",      "text": "太早了，有没有八点以后的？[THINK]"},
  {"role": "assistant", "text": "有，我再帮你筛一下，稍等。"},
  {"role": "assistant", "text": "「G13次，08:05发，13:08到上海虹桥，二等座余票充足，票价553元」八点以后最早是 G13，八点零五发，下午一点零^八到[CUT]上海虹桥，票价一样五百五十三块。"},
  {"role": "user",      "text": "等等，G1 几点到来着？"},
  {"role": "assistant", "text": "G1 是十一点二十八，G13 是下午一点零八，晚将近两个小时。"},
  {"role": "user",      "text": "[PEND2S]行，那订 G13 吧，二等座，帮我订一下。[THINK]"},
  {"role": "assistant", "text": "好，我来帮你订，稍等。"},
  {"role": "assistant", "text": "「G13次明日二等座订票成功，08:05发，13:08到，票价553元，订单号XG20260604001」订好了，G13 二等座，明早八点零五发，票价五百五十三，订单号 XG20260604001，出发前提前到站取票。"}
]
```

#### 标注解析

- **`[THINK]`（第1、6、11条）**：User 每次提出需要查询或操作的新请求，在该条 User 消息末尾加 `[THINK]`。
- **`「」`（第3、8、13条）**：S2 返回结果后的第一条 assistant 必须先有 `「」`，内容是完整事实摘要；紧接着的正文只能转述 `「」` 里已有的信息，不能新增。`[THINK]` 之后的第一条 assistant 是无 `「」` 的承接句，S2 还没返回。
- **`^`/`[CUT]`（第3条）**：剧本写"[当 Agent 刚刚说完 '十一点二十' 的时候打断]"，`^` 插在"二十"之后、`[CUT]` 插在其后第1字"八"之后，`[CUT]` 后的"分到上海虹桥，全程五小时二十分，二等座五百五十三块"是幽灵文本保留在本条，下一条 assistant 全新开口不重复。
- **`^`/`[CUT]`（第8条）**：剧本写"[当 Agent 刚刚说完 '一点零' 的时候打断]"，`^` 插在"零"之后、`[CUT]` 插在其后第2字"到"之后，幽灵文本"上海虹桥，票价一样五百五十三块"保留在本条。
- **无 `[WAIT]`（第9条）**：User 追问的是已经返回过的信息，S2 无需重置，不加 `[WAIT]`，Agent 直接口头回答。
- **`[PEND2S]`（第11条）**：剧本写"[User 沉默了 2 秒后]"，转换为 `[PEND2S]` 放在该条 User 消息开头，沉默发生在 Agent 给出对比信息、User 自己权衡决策的时候。
"""


DEFAULT_API_KEY = "EMPTY"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_DEEPSEEK_LOCAL_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_DIRECTOR_PROVIDER_API_BASES = (
    "local=http://localhost:8000/v1"
)
DEFAULT_MODEL_NAME = (
    DEFAULT_DEEPSEEK_LOCAL_MODEL
)

DEFAULT_SEED_PATH = (
    "data/seed/samples.cleaned.jsonl"
)
DEFAULT_INBOUND_PATH = (
    "data/content/train_sft.voiceagent.inbound.jsonl"
)
DEFAULT_OUTBOUND_PATH = (
    "data/content/train_sft.voiceagent.outbound.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/pipeline_text"
)

DEFAULT_SAVE_INTERVAL = 2000
DEFAULT_MAX_WORKERS = 5
DEFAULT_RPM_LIMIT = 10
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY_BASE = 2
DEFAULT_SEED = 42
DEFAULT_WRITER_SEED_SAMPLE_SIZE = 3
DEFAULT_WRITER_PROVIDER_MODELS = DEFAULT_DEEPSEEK_LOCAL_MODEL
DEFAULT_WRITER_PROVIDER_API_BASES = DEFAULT_DIRECTOR_PROVIDER_API_BASES
DEFAULT_WRITER_PROVIDER_WORKERS = 2000
DEFAULT_WRITER_PROVIDER_RPM = 0
DEFAULT_WRITER_PROVIDER_QUIET_WINDOW = 60.0
DEFAULT_DIRECTOR_PROVIDER_MODELS = DEFAULT_DEEPSEEK_LOCAL_MODEL
DEFAULT_DIRECTOR_PROVIDER_WORKERS = 800
DEFAULT_DIRECTOR_PROVIDER_RPM = 0
DEFAULT_DIRECTOR_PROVIDER_QUIET_WINDOW = 60.0


@dataclass
class PipelineConfig:
    api_key: str
    api_base: str
    model_name: str
    seed_path: str
    inbound_path: str
    outbound_path: str
    output_dir: str
    mode: str
    split: str
    debug_limit: Optional[int]
    save_interval: int
    max_workers: int
    rpm_limit: int
    max_retries: int
    retry_delay_base: int
    random_seed: int
    writer_seed_sample_size: int
    writer_provider_models: str
    writer_provider_api_bases: str
    writer_provider_workers: int
    writer_provider_rpm: int
    writer_provider_quiet_window: float
    director_provider_models: str
    director_provider_api_bases: str
    director_provider_workers: int
    director_provider_rpm: int
    director_provider_quiet_window: float


class RateLimiter:
    """线程安全的限流器，确保全局请求速率不超过设定值。"""

    def __init__(self, rpm: int):
        self.interval = 0.0 if rpm <= 0 else 60.0 / rpm
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait_for_token(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            wait_time = self.interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call_time = time.time()


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="统一编剧/导演流水线")
    parser.add_argument(
        "--mode",
        choices=["writer_only", "director_only", "full_pipeline"],
        default="full_pipeline",
        help="运行模式",
    )
    parser.add_argument(
        "--split",
        choices=["inbound", "outbound", "both"],
        default="both",
        help="处理数据 split",
    )
    parser.add_argument("--debug-limit", type=int, default=None, help="仅处理前 N 条待执行样本")
    parser.add_argument(
        "--writer-seed-sample-size",
        type=int,
        default=DEFAULT_WRITER_SEED_SAMPLE_SIZE,
        help="每条 content 展开生成多少条 (content, seed) 样本",
    )
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--rpm-limit", type=int, default=DEFAULT_RPM_LIMIT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay-base", type=int, default=DEFAULT_RETRY_DELAY_BASE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--writer-provider-models",
        default=DEFAULT_WRITER_PROVIDER_MODELS,
        help=(
            "writer 阶段模型列表，格式 name=model,name=model。"
            "默认与 director 当前 DeepSeek local 模型一致。"
        ),
    )
    parser.add_argument(
        "--writer-provider-api-bases",
        default=DEFAULT_WRITER_PROVIDER_API_BASES,
        help=(
            "writer 阶段出水口地址列表，格式 name=url,name=url。"
            "默认与 director 当前 8 个 DeepSeek 出水口一致。"
        ),
    )
    parser.add_argument(
        "--writer-provider-workers",
        type=int,
        default=DEFAULT_WRITER_PROVIDER_WORKERS,
        help="writer 阶段每个出水口的 worker 数。",
    )
    parser.add_argument(
        "--writer-provider-rpm",
        type=int,
        default=DEFAULT_WRITER_PROVIDER_RPM,
        help="writer 阶段每个出水口的独立 RPM 限制。",
    )
    parser.add_argument(
        "--writer-provider-quiet-window",
        type=float,
        default=DEFAULT_WRITER_PROVIDER_QUIET_WINDOW,
        help="同一 writer 出水口近 N 秒没有 429 后，已暂停 worker 才允许恢复。",
    )
    parser.add_argument(
        "--director-provider-models",
        default=DEFAULT_DIRECTOR_PROVIDER_MODELS,
        help=(
            "director 阶段模型列表，格式 name=model,name=model。"
            "DeepSeek local 默认单出水口高并发。"
        ),
    )
    parser.add_argument(
        "--director-provider-api-bases",
        default=DEFAULT_DIRECTOR_PROVIDER_API_BASES,
        help=(
            "director 阶段出水口地址列表，格式 name=url,name=url。"
            "每个地址独立启动 director-provider-workers 个 worker。"
        ),
    )
    parser.add_argument(
        "--director-provider-workers",
        type=int,
        default=DEFAULT_DIRECTOR_PROVIDER_WORKERS,
        help="director 阶段每个出水口的 worker 数。",
    )
    parser.add_argument(
        "--director-provider-rpm",
        type=int,
        default=DEFAULT_DIRECTOR_PROVIDER_RPM,
        help="director 阶段每个出水口的独立 RPM 限制。",
    )
    parser.add_argument(
        "--director-provider-quiet-window",
        type=float,
        default=DEFAULT_DIRECTOR_PROVIDER_QUIET_WINDOW,
        help="同一出水口近 N 秒没有 429 后，已暂停 worker 才允许恢复。",
    )
    parser.add_argument("--seed-path", default=DEFAULT_SEED_PATH)
    parser.add_argument("--inbound-path", default=DEFAULT_INBOUND_PATH)
    parser.add_argument("--outbound-path", default=DEFAULT_OUTBOUND_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return PipelineConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model_name=args.model_name,
        seed_path=args.seed_path,
        inbound_path=args.inbound_path,
        outbound_path=args.outbound_path,
        output_dir=args.output_dir,
        mode=args.mode,
        split=args.split,
        debug_limit=args.debug_limit,
        save_interval=args.save_interval,
        max_workers=args.max_workers,
        rpm_limit=args.rpm_limit,
        max_retries=args.max_retries,
        retry_delay_base=args.retry_delay_base,
        random_seed=args.random_seed,
        writer_seed_sample_size=args.writer_seed_sample_size,
        writer_provider_models=args.writer_provider_models,
        writer_provider_api_bases=args.writer_provider_api_bases,
        writer_provider_workers=args.writer_provider_workers,
        writer_provider_rpm=args.writer_provider_rpm,
        writer_provider_quiet_window=args.writer_provider_quiet_window,
        director_provider_models=args.director_provider_models,
        director_provider_api_bases=args.director_provider_api_bases,
        director_provider_workers=args.director_provider_workers,
        director_provider_rpm=args.director_provider_rpm,
        director_provider_quiet_window=args.director_provider_quiet_window,
    )


def create_api_caller(config: PipelineConfig) -> "APICaller":
    try:
        from .api_client import APICaller as Caller
    except ImportError:
        try:
            from api_client import APICaller as Caller
        except ImportError as exc:
            raise RuntimeError(
                "无法导入 APICaller。请确认从脚本目录运行，且已安装 openai 依赖。"
            ) from exc
    return Caller(config.api_key, config.api_base, config.model_name)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return []

    if Path(path).suffix == ".json":
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(f"{path} 不是 JSON 列表")
        return data

    records: List[Dict[str, Any]] = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        if output_path.suffix == ".json":
            json.dump(list(records), f, ensure_ascii=False, indent=2)
        else:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def local_rng(base_seed: int, key: str) -> random.Random:
    return random.Random(base_seed + stable_hash(key) % (10**12))


def ordered_unique(items: Iterable[Any]) -> List[Any]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def choose_representative(values: Sequence[Optional[str]]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    best_value = cleaned[0]
    best_count = counts[best_value]
    for value in cleaned:
        count = counts[value]
        if count > best_count:
            best_value = value
            best_count = count
    return best_value


def flatten_messages_for_reference(messages: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            label = "Agent"
        elif role == "user":
            label = "User"
        else:
            label = role.capitalize() or "Unknown"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def normalize_role(role: str) -> Optional[str]:
    role_lower = role.strip().lower()
    if role_lower in {"assistant", "agent"}:
        return "assistant"
    if role_lower == "user":
        return "user"
    return None


def clean_json_string(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end != -1:
        return text[start:end]
    return text


PEND_PATTERN = re.compile(r"\[PEND(\d+)S\]")
S1_FORBIDDEN_DIRECTOR_PATTERN = re.compile(r"\[(?:THINK|WAIT)\]|[「」]", re.IGNORECASE)
S2_GUIDANCE_TEXT_PATTERN = re.compile(
    r"S2|\[THINK\]|\[WAIT\]|「|」|内部查询|内部推理|查询/推理|推理系统|S2系统|条子|等待感|重试感|失败感|S2延迟|高延迟|中延迟|低延迟|故障无返回|无返回|碎嘴|出消息",
    re.IGNORECASE,
)
HARDCODED_GUIDANCE_FIELDS = {
    "s2_participation_level",
    "s2_message_mode",
    "s2_verbosity_level",
    "s2_delay_level",
    "s2_no_response_recovery",
}
GUIDANCE_SUMMARY_FIELD_MAP = {
    "user_style": ("user_style", False),
    "user_opening": ("user_opening", False),
    "assistant_style": ("assistant_style", False),
    "assistant_opening": ("assistant_opening", False),
    "primary_interaction_type": ("primary_interaction_type", False),
    "secondary_interaction_types": ("secondary_interaction_type", True),
    "required_events": ("required_events", True),
    "target_turn_range": ("target_turn_range", False),
    "must_end_with_assistant": ("must_end_with_assistant", False),
    "forbidden_patterns": ("forbidden_patterns", True),
}
GUIDANCE_SECTION_TITLE = "【当前样本的特殊场景补充】"


def load_guidance_config(seed_path: str) -> Dict[str, Any]:
    guidance_path = Path(seed_path).parent / "guidance.json"
    if not guidance_path.exists():
        raise FileNotFoundError(f"未找到 guidance 配置文件: {guidance_path}")
    raw_text = guidance_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"guidance 配置文件为空: {guidance_path}")
    data = json.loads(raw_text)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{guidance_path} 不是非空 JSON 对象")
    return data


def normalize_guidance_entries(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        entries = []
        for item in value:
            text = str(item).strip()
            if text:
                entries.append(text)
        return entries
    return []


def lookup_guidance_entries(
    guidance_config: Dict[str, Any],
    field_key: str,
    field_value: str,
    guidance_type: str,
) -> List[str]:
    if not field_value or field_key in HARDCODED_GUIDANCE_FIELDS:
        return []
    value_block = (
        guidance_config.get("field_guidances", {})
        .get(field_key, {})
        .get(field_value, {})
    )
    if not isinstance(value_block, dict):
        return []
    return normalize_guidance_entries(value_block.get(guidance_type))


def build_extra_guidance_lines(
    seed_summary: Dict[str, Any],
    guidance_type: str,
    allow_s2_guidance: bool = True,
) -> List[str]:
    guidance_config = seed_summary.get("guidance_config")
    if not isinstance(guidance_config, dict) or not guidance_config:
        raise ValueError(
            f"seed_summary 缺少 guidance_config，无法构造 {guidance_type} prompt"
        )

    lines = normalize_guidance_entries(guidance_config.get(f"{guidance_type}_global"))
    for summary_key, (field_key, is_multi) in GUIDANCE_SUMMARY_FIELD_MAP.items():
        raw_value = seed_summary.get(summary_key)
        if is_multi:
            values = raw_value if isinstance(raw_value, list) else []
        else:
            values = [raw_value] if raw_value else []
        for value in values:
            lines.extend(
                lookup_guidance_entries(guidance_config, field_key, str(value), guidance_type)
            )

    deduped_lines = ordered_unique(line for line in lines if line)
    if not allow_s2_guidance:
        deduped_lines = [
            line for line in deduped_lines
            if not S2_GUIDANCE_TEXT_PATTERN.search(line)
        ]
    if not deduped_lines:
        seed_ids = seed_summary.get("seed_request_ids") or []
        raise ValueError(
            f"guidance_config 未产出 {guidance_type} 侧提示，seed_request_ids={seed_ids}"
        )
    return deduped_lines


def format_extra_guidance_block(lines: Sequence[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def assert_guidance_in_prompt(
    messages: Sequence[Dict[str, str]],
    guidance_lines: Sequence[str],
    guidance_type: str,
) -> None:
    if len(messages) < 2 or messages[1].get("role") != "user":
        raise ValueError(f"{guidance_type} prompt 缺少 user 消息，无法校验 guidance 注入")
    prompt_text = str(messages[1].get("content", ""))
    if GUIDANCE_SECTION_TITLE not in prompt_text:
        raise ValueError(f"{guidance_type} prompt 缺少特殊场景补充区块")
    for line in guidance_lines:
        expected_line = f"- {line}"
        if expected_line not in prompt_text:
            raise ValueError(f"{guidance_type} prompt 缺少 guidance 文本: {expected_line}")


def load_content_items(path: str, split: str) -> List[Dict[str, Any]]:
    records = load_jsonl(path)
    items: List[Dict[str, Any]] = []
    for row in records:
        messages = row.get("messages", [])
        if not messages:
            continue
        system_message = messages[0]
        if str(system_message.get("role", "")).lower() != "system":
            continue
        dialogue_messages = messages[1:]
        first_role = str(dialogue_messages[0].get("role", "")).lower() if dialogue_messages else ""
        expected_first_role = "assistant" if split == "outbound" else "user"
        if first_role != expected_first_role:
            continue

        prompt_id = str(row.get("prompt_id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        sample_id = f"{split}:{prompt_id}"
        items.append(
            {
                "sample_id": sample_id,
                "split": split,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "original_system": str(system_message.get("content", "")).strip(),
                "original_messages": dialogue_messages,
                "first_role": first_role,
            }
        )
    return items


def load_seed_items(path: str) -> List[Dict[str, Any]]:
    all_seeds = []
    for p in path.split(","):
        p = p.strip()
        if not p:
            continue
        guidance_config = load_guidance_config(p)
        for row in load_jsonl(p):
            cleaned = row.get("cleaned")
            if not isinstance(cleaned, dict):
                continue
            request = cleaned.get("剧本请求")
            if not isinstance(request, dict):
                continue
            all_seeds.append(
                {
                    "request_id": row.get("request_id") or request.get("请求ID") or "",
                    "request": request,
                    "guidance_config": guidance_config,
                }
            )
    return all_seeds


def build_seed_summary(sampled_seeds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    requests = [seed["request"] for seed in sampled_seeds]

    def person(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("人物风格包", {}) or {}

    def interaction(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("交互包", {}) or {}

    def s2(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("S2策略包", {}) or {}

    def constraint(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("生成约束", {}) or {}

    assistant_style = choose_representative([person(req).get("助手风格") for req in requests])
    assistant_opening = choose_representative([person(req).get("助手开场") for req in requests])
    user_style = choose_representative([person(req).get("用户风格") for req in requests])
    user_opening = choose_representative([person(req).get("用户开场") for req in requests])
    primary_interaction = choose_representative([interaction(req).get("主交互类型") for req in requests])
    secondary_interactions = ordered_unique(
        interaction(req).get("次交互类型")
        for req in requests
        if interaction(req).get("次交互类型")
    )
    required_events = ordered_unique(
        event
        for req in requests
        for event in (interaction(req).get("必须出现的事件") or [])
        if event
    )
    s2_participation = choose_representative([s2(req).get("S2参与程度") for req in requests])
    s2_message_mode = choose_representative([s2(req).get("S2出消息方式") for req in requests])
    s2_verbosity = choose_representative([s2(req).get("S2碎嘴程度") for req in requests])
    s2_delay = choose_representative([s2(req).get("S2延迟程度") for req in requests])
    s2_no_response_recovery = choose_representative(
        [s2(req).get("S2无返回时纠错方式") for req in requests]
    )
    target_turn_range = choose_representative([constraint(req).get("目标轮数范围") for req in requests])
    must_end_with_assistant = choose_representative(
        [constraint(req).get("必须以助手结尾") for req in requests]
    )
    forbidden_patterns = ordered_unique(
        pattern
        for req in requests
        for pattern in (constraint(req).get("禁止出现") or [])
        if pattern
    )

    return {
        "seed_request_ids": [seed["request_id"] for seed in sampled_seeds],
        "assistant_style": assistant_style,
        "assistant_opening": assistant_opening,
        "user_style": user_style,
        "user_opening": user_opening,
        "primary_interaction_type": primary_interaction,
        "secondary_interaction_types": secondary_interactions,
        "required_events": required_events,
        "s2_participation_level": s2_participation,
        "s2_message_mode": s2_message_mode,
        "s2_verbosity_level": s2_verbosity,
        "s2_delay_level": s2_delay,
        "s2_no_response_recovery": s2_no_response_recovery,
        "target_turn_range": target_turn_range,
        "must_end_with_assistant": must_end_with_assistant,
        "forbidden_patterns": forbidden_patterns,
        "seed_snapshots": requests,
    }


def with_runtime_guidance(
    seed_summary: Dict[str, Any], guidance_config: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(guidance_config, dict) or not guidance_config:
        raise ValueError("guidance_config 不能为空，无法注入运行时 prompt")
    if "guidance_config" in seed_summary:
        raise ValueError("seed_summary 不应持久化 guidance_config")
    enriched = dict(seed_summary)
    enriched["guidance_config"] = guidance_config
    return enriched


def build_writer_system(content_item: Dict[str, Any], seed_summary: Dict[str, Any]) -> str:
    base_system = content_item["original_system"].strip() or "你是有用的助手。"
    # SI 场景：system 已经是 SI 专用格式，不追加风格字段
    if "同声传译" in base_system:
        return base_system
    additions = []
    if seed_summary.get("assistant_style"):
        additions.append(f"你的助手风格是：{seed_summary['assistant_style']}。")
    if seed_summary.get("assistant_opening"):
        additions.append(f"你的开场方式要求：{seed_summary['assistant_opening']}。")

    return "\n".join([base_system] + additions).strip()


def is_s2_related_text(value: Any) -> bool:
    return bool(S2_GUIDANCE_TEXT_PATTERN.search(str(value)))


def filter_s2_related_entries(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value and not is_s2_related_text(value)]


def strip_s2_from_seed_request(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, child in value.items():
            if key == "S2策略包" or is_s2_related_text(key):
                continue
            stripped = strip_s2_from_seed_request(child)
            if stripped is None:
                continue
            cleaned[key] = stripped
        return cleaned
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            stripped = strip_s2_from_seed_request(item)
            if stripped is None:
                continue
            cleaned_items.append(stripped)
        return cleaned_items
    if isinstance(value, str) and is_s2_related_text(value):
        return None
    return value


def format_seed_constraints(
    seed_summary: Dict[str, Any],
    allow_s2_fields: bool = True,
) -> str:
    lines = []
    if seed_summary.get("user_style"):
        lines.append(f"- 用户风格：{seed_summary['user_style']}")
    if seed_summary.get("user_opening"):
        lines.append(f"- 用户开场：{seed_summary['user_opening']}")
    if seed_summary.get("primary_interaction_type"):
        lines.append(f"- 主交互类型：{seed_summary['primary_interaction_type']}")
    if seed_summary.get("secondary_interaction_types"):
        secondary = seed_summary["secondary_interaction_types"]
        if not allow_s2_fields:
            secondary = filter_s2_related_entries(secondary)
        if secondary:
            lines.append("- 次交互变化：" + "、".join(secondary))
    if seed_summary.get("required_events"):
        required_events = seed_summary["required_events"]
        if not allow_s2_fields:
            required_events = filter_s2_related_entries(required_events)
        if required_events:
            lines.append("- 必须出现的事件：" + "、".join(required_events))

    if allow_s2_fields:
        s2_parts = []
        if seed_summary.get("s2_participation_level"):
            s2_parts.append(f"S2参与程度={seed_summary['s2_participation_level']}")
        if seed_summary.get("s2_message_mode"):
            s2_parts.append(f"S2出消息方式={seed_summary['s2_message_mode']}")
        if seed_summary.get("s2_verbosity_level"):
            s2_parts.append(f"S2碎嘴程度={seed_summary['s2_verbosity_level']}")
        if seed_summary.get("s2_delay_level"):
            s2_parts.append(f"S2延迟程度={seed_summary['s2_delay_level']}")
        if seed_summary.get("s2_no_response_recovery"):
            s2_parts.append(
                f"S2无返回时纠错方式={seed_summary['s2_no_response_recovery']}"
            )
        if s2_parts:
            lines.append("- S2策略： " + "；".join(s2_parts))

    if seed_summary.get("target_turn_range"):
        lines.append(f"- 目标轮数范围：{seed_summary['target_turn_range']}")
    if seed_summary.get("must_end_with_assistant"):
        lines.append(
            f"- 必须以助手结尾：{seed_summary['must_end_with_assistant']}"
        )
    if seed_summary.get("forbidden_patterns"):
        forbidden_patterns = seed_summary["forbidden_patterns"]
        if not allow_s2_fields:
            forbidden_patterns = filter_s2_related_entries(forbidden_patterns)
        if forbidden_patterns:
            lines.append("- 禁止出现：" + "、".join(forbidden_patterns))

    if not allow_s2_fields:
        lines = [line for line in lines if not is_s2_related_text(line)]
    return "\n".join(lines)


def format_raw_seed_block(
    seed_summary: Dict[str, Any],
    allow_s2_fields: bool = True,
) -> str:
    seed_snapshots = seed_summary.get("seed_snapshots", [])
    if not allow_s2_fields:
        seed_snapshots = [strip_s2_from_seed_request(item) for item in seed_snapshots]
    payload = {
        "seed_request_ids": seed_summary.get("seed_request_ids", []),
        "seed_snapshots": seed_snapshots,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def describe_seed_fields(allow_s2_fields: bool = True) -> str:
    common = """
- `人物风格包.用户风格`：面向编剧和导演。描述用户在对话中的脾气、表达方式、耐心程度和情绪基调；编剧要把它写进用户台词，导演不能把它标成额外知识，只能忠实演绎。
- `人物风格包.用户开场`：面向编剧和导演。描述用户第一轮或前几轮的开口方式，影响开场是否寒暄、是否直接切题。
- `人物风格包.助手风格`：面向编剧和导演。描述助手稳定的人设与说话风格，应该长期一致地体现在语气、措辞、安抚方式和推进方式里。
- `人物风格包.助手开场`：面向编剧和导演。描述助手的起手方式，主要影响第一句或前几句的开场口吻。
- `交互包.主交互类型`：面向编剧和导演。描述整段对话最主要的推进模式，决定剧情主线怎么往前走。
- `交互包.次交互类型`：面向编剧和导演。描述主线之外的次级变化，用来制造改口、补充条件、转折、追问或节奏变化。
- `交互包.必须出现的事件`：面向编剧和导演。描述剧情里必须实际发生的关键节点，不能漏掉。
- `生成约束.目标轮数范围`：面向编剧和导演。描述整段对话大致应该有多少轮，影响长度和节奏密度。
- `生成约束.必须以助手结尾`：面向编剧和导演。描述结尾角色约束，决定最后一轮应该由谁收尾。
- `生成约束.禁止出现`：面向编剧和导演。描述全局禁忌模式，是编剧和导演都必须严格规避的错误行为。
""".strip()
    if not allow_s2_fields:
        return common
    s2 = """
- `S2策略包.S2参与程度`：面向编剧和导演。它决定这段对话有多依赖查询、核查、计算、操作或推理。
  `关`：尽量不要把任务写成强事实核查型、强检索型、强计算型问题，更像普通闲聊、轻陪伴、浅建议、常识性回应。
  `低`：只允许少量查询、少量计算、少量核查或简单操作，S2 不是主线。
  `中`：需要中等强度的查询、核查、计算或操作，S2 会多次参与，但不能铺天盖地。
  `高`：需要高强度、较难或多步骤的问题求解、核查或操作，S2 是支撑回答的重要部分。
- `S2策略包.S2出消息方式`：主要面向导演，也给编剧作为节奏参考。
  `大段返回`：内部信息更适合成块到达，导演应让 `「...」` 相对完整、成段，减少过碎切分。
  `小段返回`：内部信息更适合分小块到达，导演应把 `「...」` 分成多次较短但仍完整的信息片段。
- `S2策略包.S2碎嘴程度`：主要面向导演，也给编剧作为表层节奏参考。
  `低频`：S2 很少出条子，导演不要频繁塞 `「...」`。
  `中频`：S2 适中参与，哪里确实需要事实、结果、条件或计算结论，哪里再出条子。
  `高频`：S2 参与比较频繁，但每条 `「...」` 仍必须完整，不能变成残缺关键词。
  `洪泛`：S2 非常密集地参与，导演可以多次插入 `「...」`，但仍要保证时序自然、信息完整、不显得乱。
- `S2策略包.S2延迟程度`：面向编剧和导演。描述内部查询/推理/操作返回速度，决定等待感如何表演。
  `低延迟`：结果回来较快，允许较快进入事实型回答，但仍要自然，不要像瞬移。
  `中延迟`：需要明显的过渡、垫话、确认或安抚，不能刚触发就立刻拿到完整结果。
  `高延迟`：要明确演出“还在查询/还在确认/请稍等”的过程，等待感更强。
  `故障无返回`：第一次尝试可能拿不到结果，要演出失败、致歉、再试一次；如果还不行，就不要编造。
- `S2策略包.S2无返回时纠错方式`：面向编剧和导演。描述查询/推理/操作迟迟没有结果时，助手应该怎样自然补救。注意是表演助手表层话术，不是把内部机制直接说出口。
""".strip()
    return common + "\n" + s2


def build_writer_task_background() -> str:
    return """
你现在在做两阶段流水线的第一阶段，也就是编剧阶段。
编剧的职责是把内容主题、system设定和 seed 约束改写成自然、可演、可被导演二次标注的纯文本剧本。
这里默认存在一个“表层助手”和“内部查询/推理系统”的双层工作机制，但编剧阶段不需要显式输出任何 S2 消息，也不要写 `「...」`、`[THINK]`、`[WAIT]`、`[CUT]` 这类导演标签。
你只需要把这种双层机制体现在剧情表面现象上：什么时候助手会先垫一句，什么时候会确认、致歉、改口、等待、重试、收尾，都应该通过自然台词和轮次节奏演出来。
如果 seed 里给了 `S2参与程度=关`，就尽量不要把问题写成强事实核查、强查询、强计算的任务，更像普通闲聊、轻陪伴或无需外部结果支撑的普通交流。
如果 seed 里给了 `S2参与程度=低/中/高`，你要把它理解为剧情对查询、核查、计算、操作或推理的依赖程度不同；强度越高，越要在表层对话里演出等待、核查、改口、重试或确认过程。
如果 seed 里给了低延迟、中延迟、高延迟、故障无返回之类的信息，你要把它理解成“助手表面上会经历等待、查询/推理/操作失败、再次尝试、无法确认、最终收束”的节奏约束，而不是把内部机制直接说出来。
""".strip()


def build_director_task_background(allow_s2_fields: bool = True) -> str:
    if not allow_s2_fields:
        return """
你现在在做两阶段流水线的第二阶段，也就是导演阶段。
编剧负责写自然剧本，导演负责把自然剧本转换成带控制标签的训练数据。
本条样本只做表层对话标注；你的职责不是改剧情，而是忠实演绎剧本，并把 seed 里的非内部系统约束精确落实到角色、轮次、打断、沉默、目标轮数和结尾角色上。
""".strip()
    return """
你现在在做两阶段流水线的第二阶段，也就是导演阶段。
编剧负责写自然剧本，导演负责把自然剧本转换成带控制标签的训练数据。
这里默认存在一个“表层助手”和“内部查询/推理系统”的双层工作机制：用户听到的是表层助手的话，`[THINK]` / `「...」` / `[WAIT]` 等标签标的是内部查询或推理的触发、返回、暂停与失败。
你的职责不是改剧情，而是忠实演绎剧本，并把 seed 里的隐藏约束精确落实到标签时序、信息到达顺序、等待感、重试感、失败感和收尾方式上。
尤其要把 seed 里与 S2 延迟、无返回、纠错方式有关的约束真正演出来，而不是只在摘要里看过却没有体现在结果里。
""".strip()


def build_delay_guidance(seed_summary: Dict[str, Any]) -> str:
    participation = seed_summary.get("s2_participation_level", "")
    message_mode = seed_summary.get("s2_message_mode", "")
    verbosity = seed_summary.get("s2_verbosity_level", "")
    delay = seed_summary.get("s2_delay_level", "")
    recovery = seed_summary.get("s2_no_response_recovery", "")
    lines = [
        "- 如果 seed 里出现 S2 延迟相关约束，你必须把“等待中的表层话术”演出来，不能让事实型回答无缘无故瞬间到达。",
        "- 所谓“承认查询/推理/操作有问题”，是指助手只能从表层角度说“我这边刚才没查到”“我这边结果还没出来”“我再帮你试一次”“抱歉刚才那次没成功”这类话，不能直接暴露自己有 S2 或内部模块。",
    ]
    if participation == "关":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“关”：尽量不要把这段对话标成强事实核查、强查询、强计算场景。除非剧本表层已经明确要求，否则不要强行插入大量 `[THINK]` 或 `「...」`，整体应更接近普通闲聊或无需依赖外部结果的自然交流。"
        )
    if participation == "低":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“低”：只在少量确实需要核查、计算、操作或查询的地方触发 `[THINK]` 和 `「...」`，不要铺太满。"
        )
    if participation == "中":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“中”：需要中等强度的查询、核查、计算或操作支撑回答，S2 会多次参与，但仍要克制、自然。"
        )
    if participation == "高":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“高”：这是高强度、较难或多步骤的问题，导演应允许更多 `[THINK]` 与 `「...」` 参与，但每次都要符合时序和因果。"
        )
    if message_mode == "大段返回":
        lines.append(
            "- 当前样本的 `S2出消息方式` 是“大段返回”：相关 `「...」` 更适合成块到达，不要切得过碎。"
        )
    if message_mode == "小段返回":
        lines.append(
            "- 当前样本的 `S2出消息方式` 是“小段返回”：相关 `「...」` 更适合分成多次较短但完整的信息片段到达。"
        )
    if verbosity == "低频":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“低频”：不要频繁插入 `「...」`，只在真正需要时出现。意思就是`「...」`要少。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。"
        )
    if verbosity == "中频":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“中频”：S2 参与频率适中，保持自然节奏，不稀也不过密。意思就是`「...」`要适中。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。"
        )
    if verbosity == "高频":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“高频”：可以较频繁插入 `「...」`，但每条都必须是完整信息，不能退化成关键词。意思就是`「...」`要比较多。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。无论S2多么嘈杂，你与用户的交互必须按照助手性格来，不能错乱。"
        )
    if verbosity == "洪泛":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“洪泛”：S2 可以非常密集地参与，但你仍必须保证信息完整、时序自然、标签不过载。意思就是`「...」`要多。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。无论S2多么嘈杂，你与用户的交互必须按照助手性格来，不能错乱。"
        )
    if delay == "低延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“低延迟”：结果可以比较快回来，但也要保留基本的自然衔接，不要让事实像零延迟瞬移出现。"
        )
    if delay == "中延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“中延迟”：首次 `[THINK]` 之后，通常要先有 1 到 2 句自然垫话、确认、安抚或过渡，再让相关 `「...」` 到达。不要让事实型内容秒回。"
        )
    if delay == "高延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“高延迟”：你必须演出明显等待感。通常要先有 1 到 2 句，表层助手应明确表示还在查询/推理/操作、还在确认、请稍等，而不是假装已经知道答案。"
        )
    if delay == "故障无返回":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“故障无返回”：第一次 `[THINK]` 后，如果没有返回，助手必须承认这次查询/推理/操作没有成功并道歉，然后再次尝试一次 `[THINK]`。"
        )
        lines.append(
            "- 如果第二次尝试后仍然没有可用返回，就不要再编造事实。助手应再次道歉，明确自己当前无法确认，并自然结束对话或结束当前轮次。"
        )
    if recovery:
        lines.append(
            f"- 当前样本的 `S2无返回时纠错方式` 是“{recovery}”：你要把这个纠错思路落实成表层助手的自然话术和收束方式。并且当故障时，不要在对话中加入`「...」`包裹的S2信息。因为S2故障了。"
        )
    return "\n".join(lines)


def build_writer_s2_guidance(seed_summary: Dict[str, Any]) -> str:
    participation = seed_summary.get("s2_participation_level", "")
    delay = seed_summary.get("s2_delay_level", "")
    recovery = seed_summary.get("s2_no_response_recovery", "")
    lines = [
        "- 编剧阶段就要先把这段对话定性清楚：它到底更像普通交流，还是少量查询型、中等查询型、高强度查询/计算/核查型。这个强度必须在剧本结构里提前决定，不能把难题留给导演硬补。",
    ]
    if participation == "关":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“关”：请把剧本写成不依赖外部查询/推理/操作结果也能自然成立的对话。优先写普通闲聊、轻咨询、浅建议、常识性说明、情绪安抚或流程性沟通，不要把核心推进建立在复杂事实核查、复杂计算或多步检索上。"
        )
    if participation == "低":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“低”：剧本里可以有少量查询、核查、计算或操作，但它不是主线。大部分轮次仍应靠自然沟通推进，只在少数节点需要助手去查一下、算一下、核一下。"
        )
    if participation == "中":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“中”：剧本应明显依赖中等强度的查询、核查、计算或操作。多个关键节点都可以需要结果支撑，但对话仍要自然，不能轮轮都在机械查询/推理/操作。"
        )
    if participation == "高":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“高”：编剧阶段就要把它写成高强度、较难或多步骤的问题场景。核心推进必须依赖多次查询、核查、计算、推理或操作；表层台词里要提前埋好等待、确认、分步推进、可能重试或改口的空间，否则导演后面圆不回来。"
        )
    if delay == "低延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“低延迟”：即使结果回来较快，编剧也要保留最基本的自然衔接，不要让回答显得像无因果瞬间知道。"
        )
    if delay == "中延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“中延迟”：编剧要预留 1 到 2 句自然垫话、确认或安抚，让后续导演有空间演出等待感。"
        )
    if delay == "高延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“高延迟”：编剧必须把“还在查、还在确认、请稍等”的表层过程写进剧情，否则导演无法自然演出高延迟。"
        )
    if delay == "故障无返回":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“故障无返回”：编剧必须提前把失败、致歉、再次尝试、再次失败后不乱说并收束的剧情空间写出来，不能把剧本写成一次查询/推理/操作就顺利给出答案。"
        )
    if recovery:
        lines.append(
            f"- 当前样本的 `S2无返回时纠错方式` 是“{recovery}”：编剧要把这种补救思路提前体现在表层剧情里，让导演只是忠实标注，而不是替你重写剧情。"
        )
    return "\n".join(lines)


def construct_writer_messages(
    content_item: Dict[str, Any],
    writer_system: str,
    seed_summary: Dict[str, Any],
) -> List[Dict[str, str]]:
    split = content_item["split"]
    reference_messages = flatten_messages_for_reference(content_item["original_messages"])
    first_speaker = "Agent" if split == "outbound" else "User"

    # 检测原始对话语言，决定剧本语言
    # 规则：中文 → 中文剧本；英文（拉丁字母为主）→ 英文剧本；其他语言 → 中文剧本
    _script_lang = "zh"
    for _m in content_item.get("original_messages", []):
        if _m.get("role") == "user":
            _t = _m.get("content", "")
            if len(_t) > 15:
                _zh = sum(1 for c in _t if "一" <= c <= "鿿")
                _ascii = sum(1 for c in _t if "a" <= c <= "z" or "A" <= c <= "Z")
                if _zh < len(_t) * 0.2 and _ascii > len(_t) * 0.5:
                    _script_lang = "en"
            break
    _lang_note = "\n【语言要求】原始对话为英文，本剧本的 User 和 Agent 台词必须全部用英文写。\n" if _script_lang == "en" else ""
    task_intro = "外呼任务" if split == "outbound" else "内呼任务"
    writer_guidance_lines = build_extra_guidance_lines(seed_summary, "writer")
    writer_extra_guidance = format_extra_guidance_block(writer_guidance_lines)
    extra_guidance_section = f"\n{GUIDANCE_SECTION_TITLE}\n{writer_extra_guidance}\n"

    user_content = f"""
请根据以下素材生成新的全双工对话剧本：

【任务背景】
{build_writer_task_background()}

【seed 字段释义】
{describe_seed_fields()}

【当前样本的编剧侧 S2 强度指导】
{build_writer_s2_guidance(seed_summary)}
{extra_guidance_section}
【写作要求】
1. 这是重写，不是照抄。保留原始内容的主题、任务目标、核心知识点或服务目标，但具体措辞、轮次组织、打断点、承接方式要重新生成。
2. 第一位说话者必须是 {first_speaker}。
3. 如果是外呼，Agent 要主动发起联系；如果是内呼，User 要先发起诉求。
4. 不要把 seed 的结构化字段名直接说出来，不要把“交互包”“人物风格包”“S2策略包”这些词写进剧本。
5. 助手不能提前知道用户脾气、未来会发生的事件或 S2 细节；这些只能体现为自然发生的剧情。
6. 如果约束之间有轻微冲突，请优先保证角色自然、首轮说话方正确、必须事件出现、剧情可演、结尾满足要求。
7. 为了保证下游导演可标注，严格使用 `[状态]` + `User/Agent: "台词"` 的格式。
8. 容易口播的英文品牌、型号、缩写请直接保留，不要为了口语化硬写成生硬的中文念法，比如 `R8 Spyder`、`RSQ`、`TTS` 没必要强行改写。
9. 如果用户沉默了一小段时间再继续说，可以使用 `[User沉默了1秒后]`、`[User沉默了2秒后]` 这类状态；这个写法只允许出现在用户轮前，不允许写 `Agent沉默了...`。
10. 当前只使用 1 条 seed 来约束当前剧本，不要把多条 seed 混合成一条样本。
11. 编剧阶段不要显式写出 S2 消息，不要输出 `^`、`「...」`、`[THINK]`、`[WAIT]`、`[CUT]`、`[PENDXS]`。如果 seed 涉及高延迟、故障无返回或重试，请只在表层对话里通过垫话、致歉、再次尝试、无法确认、自然收尾来体现。
12. **用户台词必须把话说完整**：User 的每一句话都要包含足够的上下文，让 Agent 能仅凭这句话就理解诉求，不能用模糊代指（"那个问题"、"刚才那个"、"能再算一遍吗"）代替实际内容，否则 Agent 的回答将成为幻觉。例如用户想重新计算某题，必须把题目条件在台词里说清楚或重复一遍，不能假设 Agent 记得之前的对话内容。
13. 如果写 `[当 Agent 刚刚说完 'XX' 的时候打断]` 或 `[当 Agent 刚刚说完 'XX' 的时候垫话]`，`XX` 必须能在上一句 Agent 台词中逐字找到，并且不能是句尾最后一个字/词；锚点后至少还要剩下两个中文可读字或一个完整英文单词。

【任务类型】
{task_intro}

【保留给助手的 system 设定】
{writer_system}

【当前这条样本对应的 seed 约束摘要】
{format_seed_constraints(seed_summary)}

【当前这条样本对应的 RAW seed】
{format_raw_seed_block(seed_summary)}

【原始内容主题】
{content_item["prompt"]}
{_lang_note}
【原始对话参考】
{reference_messages}
"""
    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]
    assert_guidance_in_prompt(messages, writer_guidance_lines, "writer")
    return messages


def construct_director_messages(
    writer_record: Dict[str, Any], s1_mode: bool = False
) -> List[Dict[str, str]]:
    split = writer_record["split"]
    first_speaker = "Agent" if split == "outbound" else "User"
    seed_summary = writer_record["seed_summary"]

    # 检测剧本语言（用比例，避免状态行中文字符的干扰）
    _script = writer_record.get("script", "")
    _zh = sum(1 for c in _script if "一" <= c <= "鿿")
    _is_en_script = len(_script) > 0 and _zh / len(_script) < 0.1
    _director_lang_note = (
        "\n【语言要求】原始剧本为英文，输出 JSON 中的 text 字段内容必须保持英文，不要翻译成中文。\n"
        if _is_en_script
        else ""
    )

    director_guidance_lines = build_extra_guidance_lines(
        seed_summary,
        "director",
        allow_s2_guidance=not s1_mode,
    )
    director_extra_guidance = format_extra_guidance_block(director_guidance_lines)
    extra_guidance_section = f"\n{GUIDANCE_SECTION_TITLE}\n{director_extra_guidance}\n"

    if s1_mode:
        # S1 与 S2 共用 director 主结构，只过滤 S2 策略包、S2 标签机制和相关 guidance。
        seed_notes = format_seed_constraints(seed_summary, allow_s2_fields=False)
        user_content = f"""
请将以下剧本转换为导演标注数据。

【任务背景】
{build_director_task_background(allow_s2_fields=False)}

【seed 字段释义】
{describe_seed_fields(allow_s2_fields=False)}

【输出要求】
1. 只输出 JSON 列表，不要输出解释文字。
2. 只输出 User / Agent 对话消息，不要输出 system。
3. 保留外呼和内呼的首轮差异：outbound 首轮必须是 Agent，inbound 首轮必须是 User。
4. 如果剧本里出现 `[User沉默了X秒后]`，请把它转换成对应 User 消息开头的 `[PENDXS]`，例如 `[User沉默了1秒后]` -> `[PEND1S]`。
5. 本条样本只做表层转写；`text` 中只允许使用 `^`、`[CUT]` 和 `[PENDXS]`，不要添加其他控制标签。
6. 导演必须保留并落实 seed 中的非内部系统约束，包括用户/助手风格、开场、主交互、次交互、必须事件、目标轮数、结尾角色和禁止模式。

【样本类型】
{split}

【首轮说话方要求】
{first_speaker}

【保留给助手的 system 设定】
{writer_record["writer_system"]}

【编剧约束摘要】
{seed_notes}

【当前这条样本对应的 RAW seed】
{format_raw_seed_block(seed_summary, allow_s2_fields=False)}
{extra_guidance_section}
【原始内容主题】
{writer_record["prompt"]}
{_director_lang_note}
【原始剧本】
{writer_record["script"]}
"""
        messages = [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT_S1},
            {"role": "user", "content": user_content.strip()},
        ]
        assert_guidance_in_prompt(messages, director_guidance_lines, "director")
        return messages

    # ── S2 完整版 user prompt ─────────────────────────────────────────────────
    seed_notes = format_seed_constraints(seed_summary)
    user_content = f"""
请将以下剧本转换为导演标注数据。

【任务背景】
{build_director_task_background()}

【seed 字段释义】
{describe_seed_fields()}

【输出要求】
1. 只输出 JSON 列表，不要输出解释文字。
2. 只输出 User / Agent 对话消息，不要输出 system。
3. 保留外呼和内呼的首轮差异：outbound 首轮必须是 Agent，inbound 首轮必须是 User。
4. 如果剧本里出现 `[User沉默了X秒后]`，请把它转换成对应 User 消息开头的 `[PENDXS]`，例如 `[User沉默了1秒后]` -> `[PEND1S]`。
5. 当上一条消息是 User 时，下一条 Assistant 绝不能以 `「` 开头；必须先给一句不带 S2 的短承接，再在后续 Assistant 消息中引入 S2。
6. 每一个 `「...」` 都必须是完整且准确的信息片段，不能只写标题、关键词或提纲；Agent 正文不能说出 `「...」` 里没有先到达的关键事实。
7. 你必须严格参考 RAW seed 中的 `S2策略包` 来决定等待感、重试感、失败感和收尾方式，不能只看一眼摘要就忽略。
8. 如果 RAW seed 指向"中延迟"，就要多给一些自然垫话和过渡；如果指向"高延迟"，就要明确演出"仍在查询/推理/操作/仍在确认"；如果指向"故障无返回"，就要演出第一次失败、道歉、再次尝试、再次失败后不乱说并结束对话的过程。
9. 编剧不显式写 S2 消息，但导演必须根据剧本表层现象和 seed 约束，把内部触发、等待、返回、失败、重试准确补出来。

【样本类型】
{split}

【首轮说话方要求】
{first_speaker}

【保留给助手的 system 设定】
{writer_record["writer_system"]}

【编剧约束摘要】
{seed_notes}

【当前这条样本对应的 RAW seed】
{format_raw_seed_block(seed_summary)}

【当前样本的 S2 延迟/失败演绎要求】
{build_delay_guidance(seed_summary)}
{extra_guidance_section}
【原始内容主题】
{writer_record["prompt"]}
{_director_lang_note}
【原始剧本】
{writer_record["script"]}
"""
    messages = [
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]
    assert_guidance_in_prompt(messages, director_guidance_lines, "director")
    return messages


# 价格常量（元/百万 tokens），与 api_client 解耦，便于集中修改
_DEFAULT_PRICE_PER_M = {
    "input": 3.0,   # 非缓存输入
    "cache": 0.025, # 缓存命中输入
    "output": 6.0,  # 输出
}
_MODEL_PRICE_PER_M = {
    "deepseek-v4-pro": _DEFAULT_PRICE_PER_M,
    "deepseek-v4-pro-baidu": _DEFAULT_PRICE_PER_M,
    "deepseek-v4-flash": {
        "input": 1.0,
        "cache": 0.02,
        "output": 2.0,
    },
    "deepseek-v4-flash-baidu": {
        "input": 1.0,
        "cache": 0.02,
        "output": 2.0,
    },
    "deepseek-v4-flash-tencent": {
        "input": 1.0,
        "cache": 0.02,
        "output": 2.0,
    },
    "deepseek-v4-flash-huawei": {
        "input": 1.0,
        "cache": 0.02,
        "output": 2.0,
    },
}


def _price_for_model(model_name: str) -> Dict[str, float]:
    normalized = str(model_name).lower()
    for key, prices in _MODEL_PRICE_PER_M.items():
        if key in normalized:
            return prices
    return _DEFAULT_PRICE_PER_M


def _calc_cost(
    prompt: int,
    completion: int,
    cached: int,
    model_name: str = DEFAULT_MODEL_NAME,
) -> float:
    """计算单次请求费用（元）"""
    prices = _price_for_model(model_name)
    non_cached = max(prompt - cached, 0)
    return (
        non_cached  / 1_000_000 * prices["input"]
        + cached    / 1_000_000 * prices["cache"]
        + completion / 1_000_000 * prices["output"]
    )


def call_api_with_retry(
    caller: "APICaller",
    limiter: RateLimiter,
    messages: List[Dict[str, str]],
    max_retries: int,
    retry_delay_base: int,
) -> Tuple[str, tuple]:
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            limiter.wait_for_token()
            raw_text, token_stats = caller.call(messages)
            if raw_text:
                return raw_text, token_stats
            raise ValueError("API returned empty response")
        except Exception as exc:  # pragma: no cover - defensive runtime path
            last_exception = exc
            if attempt < max_retries:
                time.sleep(retry_delay_base * (2 ** attempt))
    raise RuntimeError(f"API 调用失败: {last_exception}")


def _strip_think_tags(text: str) -> str:
    cleaned = re.sub(
        r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_usage_detail(usage: Any) -> Dict[str, int]:
    cached_tokens = 0
    reasoning_tokens = 0
    cache_write_tokens = 0

    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cached_tokens = getattr(ptd, "cached_tokens", None) or 0

    extra = getattr(usage, "model_extra", None) or {}
    if cached_tokens == 0:
        cached_tokens = (
            extra.get("cache_read_tokens")
            or extra.get("cached_tokens")
            or extra.get("effectiveCachedTokens")
            or 0
        )

    ctd = getattr(usage, "completion_tokens_details", None)
    if ctd is not None:
        reasoning_tokens = getattr(ctd, "reasoning_tokens", None) or 0

    cache_write_tokens = extra.get("cache_write_tokens", 0) or 0
    return {
        "cached_tokens": int(cached_tokens),
        "reasoning_tokens": int(reasoning_tokens),
        "cache_write_tokens": int(cache_write_tokens),
    }


def _usage_to_token_stats(usage: Any) -> Tuple[int, int, int, Dict[str, int]]:
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return completion_tokens, prompt_tokens, total_tokens, _extract_usage_detail(usage)


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 429 or "429" in text or "rate limit" in text


def _is_provider_fatal_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code in {401, 403, 404}


def _is_drop_record_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 400 or "status_code: 400" in text or " 400 " in text


def parse_named_models(raw: str, stage: str) -> List[Tuple[str, str]]:
    providers: List[Tuple[str, str]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            name, model = item.split("=", 1)
            name = name.strip()
            model = model.strip()
        else:
            model = item
            name = model.rsplit("-", 1)[-1].strip() or model
        if not name or not model:
            raise ValueError(f"非法 {stage} provider 配置: {part!r}")
        providers.append((name, model))
    if not providers:
        raise ValueError(f"{stage} provider 列表为空")
    return providers


def parse_named_api_bases(raw: str, stage: str) -> List[Tuple[str, str]]:
    providers: List[Tuple[str, str]] = []
    for idx, part in enumerate(raw.split(",")):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            name, api_base = item.split("=", 1)
            name = name.strip()
            api_base = api_base.strip()
        else:
            name = f"provider_{idx}"
            api_base = item
        if not name or not api_base:
            raise ValueError(f"非法 {stage} api_base 配置: {part!r}")
        providers.append((name, api_base))
    if not providers:
        raise ValueError(f"{stage} api_base 列表为空")
    return providers


def resolve_provider_specs(
    provider_models: str,
    provider_api_bases: str,
    stage: str,
) -> List[Tuple[str, str, str]]:
    api_base_specs = parse_named_api_bases(provider_api_bases, stage)
    model_specs = parse_named_models(provider_models, stage)
    seen_names: set[str] = set()
    for name, _ in api_base_specs:
        if name in seen_names:
            raise ValueError(f"{stage} api_base 出水口名重复: {name}")
        seen_names.add(name)

    if len(model_specs) == 1:
        model_name = model_specs[0][1]
        return [(name, model_name, api_base) for name, api_base in api_base_specs]

    model_by_name = {name: model_name for name, model_name in model_specs}
    if len(model_by_name) != len(model_specs):
        raise ValueError(f"{stage} provider model 名称重复")

    if all(name in model_by_name for name, _ in api_base_specs):
        return [
            (name, model_by_name[name], api_base)
            for name, api_base in api_base_specs
        ]

    if len(model_specs) == len(api_base_specs):
        return [
            (api_name, model_name, api_base)
            for (api_name, api_base), (_, model_name) in zip(api_base_specs, model_specs)
        ]

    raise ValueError(
        f"{stage} provider model 数量与 api_base 数量不匹配；"
        "请只给一个模型，或按 api_base 名称逐一配置模型"
    )


def resolve_director_provider_specs(config: PipelineConfig) -> List[Tuple[str, str, str]]:
    return resolve_provider_specs(
        config.director_provider_models,
        config.director_provider_api_bases,
        "director",
    )


def resolve_writer_provider_specs(config: PipelineConfig) -> List[Tuple[str, str, str]]:
    return resolve_provider_specs(
        config.writer_provider_models,
        config.writer_provider_api_bases,
        "writer",
    )


class ProviderSendLimiter:
    """同一出水口限速器；rpm<=0 表示本地 vLLM 不限速。"""

    def __init__(self, rpm: int):
        self.interval = 0.0 if rpm <= 0 else 60.0 / rpm
        self.next_send_at = 0.0
        self.lock = threading.Lock()

    def wait_for_slot(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.time()
            wait_time = self.next_send_at - now
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()
            self.next_send_at = max(now, self.next_send_at) + self.interval


@dataclass
class DirectorProviderState:
    name: str
    model_name: str
    api_key: str
    api_base: str
    rpm: int
    client: Any = field(init=False)
    http_client: Any = field(init=False)
    limiter: ProviderSendLimiter = field(init=False)
    last_429_at: float = 0.0
    next_worker_resume_at: float = 0.0
    disabled: bool = False
    disable_reason: str = ""
    stats: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
            import httpx
        except ImportError as exc:
            raise RuntimeError("无法导入 openai/httpx，director 多 provider 调度不可用") from exc
        timeout = httpx.Timeout(
            timeout=99999.0,
            connect=99999.0,
            read=99999.0,
            write=99999.0,
            pool=99999.0,
        )
        limits = httpx.Limits(
            max_connections=2000,
            max_keepalive_connections=2000,
            keepalive_expiry=30.0,
        )
        self.http_client = httpx.Client(timeout=timeout, limits=limits, trust_env=False)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            max_retries=0,
            http_client=self.http_client,
        )
        self.limiter = ProviderSendLimiter(self.rpm)

    def mark_429(self) -> None:
        with self.lock:
            self.last_429_at = time.time()
            self.stats["429"] += 1

    def mark_success(self, token_stats: Tuple[int, int, int, Dict[str, int]]) -> None:
        completion, prompt, _, detail = token_stats
        with self.lock:
            self.stats["success"] += 1
            self.stats["prompt_tokens"] += prompt
            self.stats["completion_tokens"] += completion
            self.stats["cached_tokens"] += int(detail.get("cached_tokens", 0))

    def mark_error(self, key: str) -> None:
        with self.lock:
            self.stats[key] += 1

    def disable(self, reason: str) -> None:
        with self.lock:
            self.disabled = True
            self.disable_reason = reason[:1000]
            self.stats["fatal_provider"] += 1

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        try:
            self.http_client.close()
        except Exception:
            pass

    def is_disabled(self) -> bool:
        with self.lock:
            return self.disabled

    def can_resume_worker(self, quiet_window: float) -> bool:
        with self.lock:
            return time.time() - self.last_429_at >= quiet_window

    def wait_for_worker_resume(self, quiet_window: float, stop_event: threading.Event) -> bool:
        """429 后逐个恢复 worker：静默 quiet_window 后放回 1 个，再等 quiet_window。"""
        while not stop_event.is_set():
            with self.lock:
                now = time.time()
                resume_at = max(self.last_429_at + quiet_window, self.next_worker_resume_at)
                if now >= resume_at:
                    self.next_worker_resume_at = now + quiet_window
                    self.stats["worker_resumed"] += 1
                    return True
                wait_time = min(resume_at - now, 0.5)
            time.sleep(max(wait_time, 0.05))
        return False

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "name": self.name,
                "model_name": self.model_name,
                "api_base": self.api_base,
                "disabled": self.disabled,
                "disable_reason": self.disable_reason,
                "last_429_at": self.last_429_at,
                "next_worker_resume_at": self.next_worker_resume_at,
                "stats": dict(self.stats),
            }


@dataclass
class DirectorTask:
    seq: int
    writer_record: Dict[str, Any]
    s1_mode: bool
    attempts: int = 0


@dataclass
class WriterTask:
    seq: int
    content_item: Dict[str, Any]
    attempts: int = 0


def build_writer_request_record(
    content_item: Dict[str, Any],
    seed_items: Sequence[Dict[str, Any]],
    config: PipelineConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    if "seed_item" in content_item:
        sampled_seeds = [content_item["seed_item"]]
    else:
        sampled_seeds = build_sampled_seeds(
            seed_items,
            config.writer_seed_sample_size,
            config.random_seed,
            content_item["sample_id"],
        )
    if not sampled_seeds:
        raise ValueError(f"样本 {content_item['sample_id']} 没有可用 seed")
    seed_summary = build_seed_summary(sampled_seeds)
    runtime_seed_summary = with_runtime_guidance(
        seed_summary,
        sampled_seeds[0]["guidance_config"],
    )
    writer_system = build_writer_system(content_item, seed_summary)
    writer_messages = construct_writer_messages(
        content_item,
        writer_system,
        runtime_seed_summary,
    )
    base_record = {
        "sample_id": content_item["sample_id"],
        "split": content_item["split"],
        "prompt_id": content_item["prompt_id"],
        "prompt": content_item["prompt"],
        "original_system": content_item["original_system"],
        "writer_system": writer_system,
        "original_messages": content_item["original_messages"],
        "seed_summary": seed_summary,
        "guidance_config": sampled_seeds[0]["guidance_config"],
        "writer_messages": writer_messages,
    }
    return base_record, writer_messages


def _call_writer_provider(
    provider: DirectorProviderState,
    messages: List[Dict[str, str]],
) -> Tuple[str, Tuple[int, int, int, Dict[str, int]]]:
    response = provider.client.chat.completions.create(
        model=provider.model_name,
        messages=messages,
        max_tokens=16384,
        temperature=0.6,
        extra_body={
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "max",
            }
        },
    )
    if not response or not response.choices:
        raise ValueError("API returned empty choices")
    message = response.choices[0].message
    raw_text = _strip_think_tags(message.content or "")
    if not raw_text:
        raise ValueError("API returned empty response")
    return raw_text, _usage_to_token_stats(response.usage)


def process_writer_record_with_provider(
    task: WriterTask,
    seed_items: Sequence[Dict[str, Any]],
    config: PipelineConfig,
    provider: DirectorProviderState,
) -> Dict[str, Any]:
    base_record, writer_messages = build_writer_request_record(
        task.content_item,
        seed_items,
        config,
    )
    raw_text, token_stats = _call_writer_provider(provider, writer_messages)
    return {
        **base_record,
        "script": raw_text.strip(),
        "prompt_tokens": token_stats[1],
        "completion_tokens": token_stats[0],
        "total_tokens": token_stats[2],
        "cached_tokens": token_stats[3].get("cached_tokens", 0),
        "cache_write_tokens": token_stats[3].get("cache_write_tokens", 0),
        "reasoning_tokens": token_stats[3].get("reasoning_tokens", 0),
        "writer_provider": provider.name,
        "writer_model_name": provider.model_name,
    }


def _call_director_provider(
    provider: DirectorProviderState,
    messages: List[Dict[str, str]],
) -> Tuple[str, Tuple[int, int, int, Dict[str, int]]]:
    response = provider.client.chat.completions.create(
        model=provider.model_name,
        messages=messages,
        max_tokens=16384,
        temperature=0.6,
        extra_body={
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "max",
            }
        },
    )
    if not response or not response.choices:
        raise ValueError("API returned empty choices")
    message = response.choices[0].message
    raw_text = _strip_think_tags(message.content or "")
    if not raw_text:
        raise ValueError("API returned empty response")
    return raw_text, _usage_to_token_stats(response.usage)


def process_director_record_with_provider(
    task: DirectorTask,
    config: PipelineConfig,
    provider: DirectorProviderState,
) -> Dict[str, Any]:
    writer_record = task.writer_record
    runtime_writer_record = dict(writer_record)
    runtime_writer_record["seed_summary"] = with_runtime_guidance(
        writer_record["seed_summary"],
        writer_record["guidance_config"],
    )
    director_messages = construct_director_messages(
        runtime_writer_record,
        s1_mode=task.s1_mode,
    )
    raw_response, token_stats = _call_director_provider(provider, director_messages)
    dialogue_messages = parse_director_output(raw_response)
    validate_director_messages(
        writer_record["split"],
        dialogue_messages,
        s1_mode=task.s1_mode,
    )
    final_messages = [
        {"role": "system", "content": writer_record["writer_system"]},
        *dialogue_messages,
    ]
    return {
        "sample_id": writer_record["sample_id"],
        "split": writer_record["split"],
        "prompt_id": writer_record["prompt_id"],
        "prompt": writer_record["prompt"],
        "messages": final_messages,
        "metadata": {
            "writer_system": writer_record["writer_system"],
            "original_system": writer_record["original_system"],
            "seed_summary": writer_record["seed_summary"],
            "writer_script": writer_record["script"],
            "writer_total_tokens": writer_record.get("total_tokens", 0),
            "director_prompt_tokens": token_stats[1],
            "director_completion_tokens": token_stats[0],
            "director_total_tokens": token_stats[2],
            "director_cached_tokens": token_stats[3].get("cached_tokens", 0),
            "director_cache_write_tokens": token_stats[3].get("cache_write_tokens", 0),
            "director_reasoning_tokens": token_stats[3].get("reasoning_tokens", 0),
            "s1_mode": task.s1_mode,
            "director_provider": provider.name,
            "director_model_name": provider.model_name,
        },
    }


def build_output_paths(config: PipelineConfig, split: str) -> Dict[str, str]:
    root = Path(config.output_dir)
    suffix = ".json" if config.debug_limit is not None else ".jsonl"
    return {
        "writer": str(root / f"{split}.writer{suffix}"),
        "director": str(root / f"{split}.director{suffix}"),
    }


def select_tasks(
    items: Sequence[Dict[str, Any]],
    completed_ids: set,
    debug_limit: Optional[int],
    random_seed: int,
    key: str,
) -> List[Dict[str, Any]]:
    pending = [item for item in items if item["sample_id"] not in completed_ids]
    if debug_limit is None or debug_limit >= len(pending):
        return pending
    rng = local_rng(random_seed, f"{key}:debug_limit")
    pending_copy = pending[:]
    rng.shuffle(pending_copy)
    return pending_copy[:debug_limit]


def build_sampled_seeds(
    seed_items: Sequence[Dict[str, Any]],
    seed_sample_size: int,
    random_seed: int,
    sample_id: str,
) -> List[Dict[str, Any]]:
    if len(seed_items) < seed_sample_size:
        raise ValueError("可用 seed 数量不足")
    rng = local_rng(random_seed, f"{sample_id}:seed_selection")
    if seed_sample_size == 1:
        return [seed_items[rng.randrange(len(seed_items))]]
    return rng.sample(seed_items, seed_sample_size)


def expand_content_items_with_seeds(
    content_items: Sequence[Dict[str, Any]],
    seed_items: Sequence[Dict[str, Any]],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    expanded_items: List[Dict[str, Any]] = []
    for content_item in content_items:
        sampled_seeds = build_sampled_seeds(
            seed_items,
            config.writer_seed_sample_size,
            config.random_seed,
            content_item["sample_id"],
        )
        for seed_item in sampled_seeds:
            seed_request_id = str(seed_item.get("request_id", "")).strip() or "seed"
            expanded_item = dict(content_item)
            expanded_item["base_sample_id"] = content_item["sample_id"]
            expanded_item["seed_item"] = seed_item
            expanded_item["sample_id"] = f"{content_item['sample_id']}::{seed_request_id}"
            expanded_items.append(expanded_item)
    return expanded_items


def process_writer_sample(
    content_item: Dict[str, Any],
    seed_items: Sequence[Dict[str, Any]],
    config: PipelineConfig,
    caller: "APICaller",
    limiter: RateLimiter,
) -> Dict[str, Any]:
    base_record, writer_messages = build_writer_request_record(
        content_item,
        seed_items,
        config,
    )
    raw_text, token_stats = call_api_with_retry(
        caller,
        limiter,
        writer_messages,
        config.max_retries,
        config.retry_delay_base,
    )
    return {
        **base_record,
        "script": raw_text.strip(),
        "prompt_tokens": token_stats[1],
        "completion_tokens": token_stats[0],
        "total_tokens": token_stats[2],
        "cached_tokens": token_stats[3].get("cached_tokens", 0) if len(token_stats) > 3 else 0,
        "cache_write_tokens": token_stats[3].get("cache_write_tokens", 0) if len(token_stats) > 3 else 0,
        "reasoning_tokens": token_stats[3].get("reasoning_tokens", 0) if len(token_stats) > 3 else 0,
    }


def parse_director_output(raw_response: str) -> List[Dict[str, str]]:
    json_str = clean_json_string(raw_response)
    data = json.loads(json_str)
    if not isinstance(data, list):
        raise ValueError("导演输出不是 JSON 列表")

    normalized = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("导演输出项不是对象")
        raw_role = str(item.get("role", ""))
        role = normalize_role(raw_role)
        text = str(item.get("text", "")).strip()
        if not role or not text:
            raise ValueError("导演输出缺少合法 role/text")
        normalized.append({"role": role, "content": text})
    if not normalized:
        raise ValueError("导演输出为空")
    return normalized


def validate_first_role(split: str, messages: Sequence[Dict[str, str]]) -> None:
    expected = "assistant" if split == "outbound" else "user"
    actual = messages[0]["role"]
    if actual != expected:
        raise ValueError(
            f"{split} 首条非 system 角色错误，期望 {expected}，实际 {actual}"
        )


def validate_director_messages(
    split: str,
    messages: Sequence[Dict[str, str]],
    s1_mode: bool = False,
) -> None:
    validate_first_role(split, messages)
    previous_role: Optional[str] = None
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()

        if role == "assistant" and PEND_PATTERN.search(content):
            raise ValueError("PEND 只能用于 User 消息，不能出现在 Assistant 消息中")

        if s1_mode and S1_FORBIDDEN_DIRECTOR_PATTERN.search(content):
            raise ValueError("S1 样本输出包含内部推理/等待标签，拒绝落盘")

        if (
            previous_role == "user"
            and role == "assistant"
            and content.startswith("「")
        ):
            raise ValueError(
                f"第 {index + 1} 条消息违反时序：User 后的第一条 Assistant 不能直接以 S2 开头"
            )

        previous_role = role


def process_director_record(
    writer_record: Dict[str, Any],
    config: PipelineConfig,
    guidance_config: Dict[str, Any],
    caller: "APICaller",
    limiter: RateLimiter,
    s1_mode: bool = False,
) -> Dict[str, Any]:
    runtime_writer_record = dict(writer_record)
    runtime_writer_record["seed_summary"] = with_runtime_guidance(
        writer_record["seed_summary"],
        guidance_config,
    )
    director_messages = construct_director_messages(runtime_writer_record, s1_mode=s1_mode)
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            raw_response, token_stats = call_api_with_retry(
                caller,
                limiter,
                director_messages,
                0,
                config.retry_delay_base,
            )
            dialogue_messages = parse_director_output(raw_response)
            validate_director_messages(
                writer_record["split"],
                dialogue_messages,
                s1_mode=s1_mode,
            )

            final_messages = [
                {"role": "system", "content": writer_record["writer_system"]},
                *dialogue_messages,
            ]
            return {
                "sample_id": writer_record["sample_id"],
                "split": writer_record["split"],
                "prompt_id": writer_record["prompt_id"],
                "prompt": writer_record["prompt"],
                "messages": final_messages,
                "metadata": {
                    "writer_system": writer_record["writer_system"],
                    "original_system": writer_record["original_system"],
                    "seed_summary": writer_record["seed_summary"],
                    "writer_script": writer_record["script"],
                    "writer_total_tokens": writer_record.get("total_tokens", 0),
                    "director_prompt_tokens": token_stats[1],
                    "director_completion_tokens": token_stats[0],
                    "director_total_tokens": token_stats[2],
                    "director_cached_tokens": token_stats[3].get("cached_tokens", 0) if len(token_stats) > 3 else 0,
                    "director_cache_write_tokens": token_stats[3].get("cache_write_tokens", 0) if len(token_stats) > 3 else 0,
                    "director_reasoning_tokens": token_stats[3].get("reasoning_tokens", 0) if len(token_stats) > 3 else 0,
                    "s1_mode": s1_mode,
                },
            }
        except Exception as exc:
            last_error = exc
            if attempt < config.max_retries:
                time.sleep(config.retry_delay_base * (2 ** attempt))

    raise RuntimeError(f"导演阶段失败: {last_error}")


def run_writer_stage(
    split: str,
    config: PipelineConfig,
    content_items: Sequence[Dict[str, Any]],
    seed_items: Sequence[Dict[str, Any]],
    caller: "APICaller",
    limiter: RateLimiter,
) -> List[Dict[str, Any]]:
    paths = build_output_paths(config, split)
    existing_records = load_jsonl(paths["writer"])
    completed_ids = {record["sample_id"] for record in existing_records}
    expanded_items = expand_content_items_with_seeds(content_items, seed_items, config)
    tasks = select_tasks(
        expanded_items,
        completed_ids,
        config.debug_limit,
        config.random_seed,
        f"{split}:writer",
    )
    if not tasks:
        print(f"[writer:{split}] 没有待处理任务")
        return existing_records

    print(
        f"[writer:{split}] 总样本 {len(expanded_items)}，"
        f"已完成 {len(completed_ids)}，待执行 {len(tasks)}"
    )
    records = existing_records[:]
    records_by_id = {record["sample_id"]: record for record in records}
    session_completed = 0
    session_total_tokens = 0
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_cached_tokens = 0
    session_cache_write_tokens = 0
    session_cost = 0.0

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                process_writer_sample,
                item,
                seed_items,
                config,
                caller,
                limiter,
            ): item["sample_id"]
            for item in tasks
        }
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Writer-{split}",
            smoothing=0,
        )
        for future in pbar:
            sample_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                pbar.set_postfix({"Fail": sample_id})
                print(f"[writer:{split}] {sample_id} 失败: {exc}")
                continue

            records_by_id[result["sample_id"]] = result
            session_completed += 1
            p  = int(result.get("prompt_tokens", 0))
            c  = int(result.get("completion_tokens", 0))
            ca = int(result.get("cached_tokens", 0))
            cw = int(result.get("cache_write_tokens", 0))
            session_prompt_tokens     += p
            session_completion_tokens += c
            session_cached_tokens     += ca
            session_cache_write_tokens += cw
            session_total_tokens      += p + c
            session_cost              += _calc_cost(p, c, ca, config.model_name)
            avg_tokens = session_total_tokens / session_completed if session_completed else 0
            pbar.set_postfix({
                "Suc": session_completed,
                "AvgTok": int(avg_tokens),
                "Cost¥": f"{session_cost:.3f}",
                "Cache%": f"{session_cached_tokens/max(session_prompt_tokens,1)*100:.0f}%",
            })

            if session_completed % config.save_interval == 0:
                save_jsonl(
                    [records_by_id[key] for key in sorted(records_by_id)],
                    paths["writer"],
                )

    final_records = [records_by_id[key] for key in sorted(records_by_id)]
    save_jsonl(final_records, paths["writer"])
    print(f"[writer:{split}] 完成后总记录数: {len(final_records)}")
    if session_completed:
        print(
            f"[writer:{split}] Token 统计 — "
            f"prompt={session_prompt_tokens:,}  "
            f"completion={session_completion_tokens:,}  "
            f"cached={session_cached_tokens:,}("
            f"{session_cached_tokens/max(session_prompt_tokens,1)*100:.1f}%)  "
            f"cache_write={session_cache_write_tokens:,}  "
            f"cost=¥{session_cost:.4f}"
        )
    return final_records


def run_writer_stage_with_provider_pool(
    split: str,
    config: PipelineConfig,
    content_items: Sequence[Dict[str, Any]],
    seed_items: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    paths = build_output_paths(config, split)
    existing_records = load_jsonl(paths["writer"])
    completed_ids = {record["sample_id"] for record in existing_records}
    expanded_items = expand_content_items_with_seeds(content_items, seed_items, config)
    tasks = select_tasks(
        expanded_items,
        completed_ids,
        config.debug_limit,
        config.random_seed,
        f"{split}:writer",
    )
    if not tasks:
        print(f"[writer:{split}] 没有待处理任务")
        return existing_records
    if config.writer_provider_workers <= 0:
        raise ValueError("writer_provider_workers 必须大于 0")
    if config.writer_provider_quiet_window < 0:
        raise ValueError("writer_provider_quiet_window 不能为负数")

    print(
        f"[writer:{split}] 总样本 {len(expanded_items)}，"
        f"已完成 {len(completed_ids)}，待执行 {len(tasks)}"
    )
    records_by_id = {record["sample_id"]: record for record in existing_records}
    session_completed = 0
    session_total_tokens = 0
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_cached_tokens = 0
    session_cache_write_tokens = 0
    session_cost = 0.0
    provider_specs = resolve_writer_provider_specs(config)
    providers = [
        DirectorProviderState(
            name=name,
            model_name=model_name,
            api_key=config.api_key,
            api_base=api_base,
            rpm=config.writer_provider_rpm,
        )
        for name, model_name, api_base in provider_specs
    ]
    task_queue: Queue[WriterTask] = Queue()
    for seq, item in enumerate(tasks):
        task_queue.put(WriterTask(seq=seq, content_item=item))
    result_queue: Queue[Tuple[str, Any]] = Queue()
    stop_event = threading.Event()
    print(
        f"[writer:{split}] provider 出水口: "
        f"{', '.join(f'{p.name}@{p.api_base}={p.model_name}' for p in providers)}；"
        f"每口 workers={config.writer_provider_workers}，"
        f"每口 rpm={config.writer_provider_rpm}"
    )

    def run_writer_task(
        provider: DirectorProviderState,
        task: WriterTask,
        worker_name: str,
    ) -> None:
        try:
            result = process_writer_record_with_provider(
                task,
                seed_items,
                config,
                provider,
            )
            provider.mark_success(
                (
                    int(result.get("completion_tokens", 0)),
                    int(result.get("prompt_tokens", 0)),
                    int(result.get("total_tokens", 0)),
                    {
                        "cached_tokens": int(result.get("cached_tokens", 0)),
                        "cache_write_tokens": int(result.get("cache_write_tokens", 0)),
                        "reasoning_tokens": int(result.get("reasoning_tokens", 0)),
                    },
                )
            )
            result_queue.put(("success", result))
        except Exception as exc:
            if _is_rate_limit_error(exc):
                provider.mark_429()
                task.attempts += 1
                if task.attempts <= config.max_retries:
                    task_queue.put(task)
                    result_queue.put(
                        (
                            "requeued",
                            {
                                "sample_id": task.content_item["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc)[:500],
                            },
                        )
                    )
                else:
                    result_queue.put(
                        (
                            "failure",
                            {
                                "sample_id": task.content_item["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc),
                            },
                        )
                    )
            elif _is_provider_fatal_error(exc):
                provider.disable(f"{type(exc).__name__}: {exc}")
                task_queue.put(task)
                result_queue.put(
                    (
                        "provider_disabled",
                        {
                            "sample_id": task.content_item["sample_id"],
                            "provider": provider.name,
                            "worker": worker_name,
                            "error": str(exc)[:500],
                        },
                    )
                )
            else:
                task.attempts += 1
                provider.mark_error("task_error")
                if task.attempts <= config.max_retries:
                    task_queue.put(task)
                    result_queue.put(
                        (
                            "retry",
                            {
                                "sample_id": task.content_item["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc)[:500],
                            },
                        )
                    )
                else:
                    result_queue.put(
                        (
                            "failure",
                            {
                                "sample_id": task.content_item["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc),
                            },
                        )
                    )

    def writer_provider_worker(provider: DirectorProviderState, worker_idx: int) -> None:
        worker_name = f"{provider.name}-{worker_idx}"
        while not stop_event.is_set() and not provider.is_disabled():
            provider.limiter.wait_for_slot()
            try:
                task = task_queue.get(timeout=0.5)
            except Empty:
                continue
            if provider.is_disabled():
                task_queue.put(task)
                return
            run_writer_task(provider, task, worker_name)

    def writer_provider_dispatcher(provider: DirectorProviderState) -> None:
        with ThreadPoolExecutor(max_workers=config.writer_provider_workers) as executor:
            futures = [
                executor.submit(writer_provider_worker, provider, worker_idx)
                for worker_idx in range(config.writer_provider_workers)
            ]
            for future in as_completed(futures):
                if stop_event.is_set() or provider.is_disabled():
                    break
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - worker guard
                    result_queue.put(
                        (
                            "worker_error",
                            {
                                "provider": provider.name,
                                "error": str(exc)[:500],
                            },
                        )
                    )

    threads: List[threading.Thread] = []
    for provider in providers:
        thread = threading.Thread(
            target=writer_provider_dispatcher,
            args=(provider,),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    failures: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(tasks), desc=f"Writer-{split}", smoothing=0)
    while session_completed + len(failures) < len(tasks):
        if all(provider.is_disabled() for provider in providers):
            stop_event.set()
            raise RuntimeError(f"[writer:{split}] 所有 provider 都已禁用，无法继续")
        try:
            event_type, payload = result_queue.get(timeout=5.0)
        except Empty:
            if task_queue.empty() and not any(thread.is_alive() for thread in threads):
                break
            continue
        if event_type == "success":
            result = payload
            records_by_id[result["sample_id"]] = result
            session_completed += 1
            p  = int(result.get("prompt_tokens", 0))
            c  = int(result.get("completion_tokens", 0))
            ca = int(result.get("cached_tokens", 0))
            cw = int(result.get("cache_write_tokens", 0))
            session_prompt_tokens     += p
            session_completion_tokens += c
            session_cached_tokens     += ca
            session_cache_write_tokens += cw
            session_total_tokens      += p + c
            model_name = str(result.get("writer_model_name") or config.model_name)
            session_cost              += _calc_cost(p, c, ca, model_name)
            avg_tokens = session_total_tokens / session_completed if session_completed else 0
            pbar.set_postfix({
                "Suc": session_completed,
                "AvgTok": int(avg_tokens),
                "Cost¥": f"{session_cost:.3f}",
                "Cache%": f"{session_cached_tokens/max(session_prompt_tokens,1)*100:.0f}%",
            })
            pbar.update(1)

            if session_completed % config.save_interval == 0:
                save_jsonl(
                    [records_by_id[key] for key in sorted(records_by_id)],
                    paths["writer"],
                )
        elif event_type == "failure":
            failures.append(payload)
            pbar.update(1)
            tqdm.write(
                f"[writer:{split}] {payload['sample_id']} 失败: "
                f"{payload['provider']} {payload['error']}"
            )
        elif event_type == "requeued":
            pbar.set_postfix({"429": payload["provider"], "Suc": session_completed})
        elif event_type == "provider_disabled":
            pbar.set_postfix({"Disable": payload["provider"], "Suc": session_completed})
            tqdm.write(
                f"[writer:{split}] provider {payload['provider']} 已禁用: "
                f"{payload['error']}"
            )
        elif event_type == "retry":
            pbar.set_postfix({"Retry": payload["provider"], "Suc": session_completed})
        elif event_type == "worker_error":
            pbar.set_postfix({"WorkerErr": payload["provider"], "Suc": session_completed})
            tqdm.write(
                f"[writer:{split}] provider {payload['provider']} worker 异常: "
                f"{payload['error']}"
            )

    stop_event.set()
    for thread in threads:
        thread.join(timeout=2.0)
    for provider in providers:
        provider.close()
    pbar.close()
    incomplete = len(tasks) - session_completed - len(failures)
    if incomplete:
        raise RuntimeError(
            f"[writer:{split}] {incomplete} 条 writer 任务未完成，拒绝静默完成"
        )

    final_records = [records_by_id[key] for key in sorted(records_by_id)]
    save_jsonl(final_records, paths["writer"])
    print(f"[writer:{split}] 完成后总记录数: {len(final_records)}")
    if failures:
        raise RuntimeError(f"[writer:{split}] {len(failures)} 条 writer 任务失败，已拒绝静默完成")
    if session_completed:
        print(
            f"[writer:{split}] Token 统计 — "
            f"prompt={session_prompt_tokens:,}  "
            f"completion={session_completion_tokens:,}  "
            f"cached={session_cached_tokens:,}("
            f"{session_cached_tokens/max(session_prompt_tokens,1)*100:.1f}%)  "
            f"cache_write={session_cache_write_tokens:,}  "
            f"cost=¥{session_cost:.4f}"
        )
        for provider in providers:
            snap = provider.snapshot()
            stats = snap["stats"]
            prompt = int(stats.get("prompt_tokens", 0))
            cached = int(stats.get("cached_tokens", 0))
            cost = _calc_cost(
                prompt,
                int(stats.get("completion_tokens", 0)),
                cached,
                snap["model_name"],
            )
            print(
                f"[writer:{split}] provider {snap['name']} — "
                f"api_base={snap['api_base']} "
                f"model={snap['model_name']} "
                f"success={stats.get('success', 0)} "
                f"429={stats.get('429', 0)} "
                f"task_error={stats.get('task_error', 0)} "
                f"paused={stats.get('worker_paused', 0)} "
                f"cache={cached/max(prompt,1)*100:.1f}% "
                f"cost=¥{cost:.4f}"
            )
    return final_records


def _load_s1s2_labels(output_dir: str, split: str) -> Dict[str, str]:
    """从 {split}.writer.s1s2.jsonl 加载 sample_id -> label 映射；缺失或非法则拒绝启动。"""
    s1s2_path = Path(output_dir) / f"{split}.writer.s1s2.jsonl"
    if not s1s2_path.exists():
        raise FileNotFoundError(
            f"[director:{split}] 缺少必需的 s1s2 label 文件: {s1s2_path}"
        )
    labels: Dict[str, str] = {}
    with open(s1s2_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            sid = str(d.get("sample_id", "")).strip()
            lbl = str(d.get("label", "")).strip().upper()
            if not sid:
                raise ValueError(f"[director:{split}] label 第 {line_no} 行缺少 sample_id")
            if lbl not in {"S1", "S2"}:
                raise ValueError(
                    f"[director:{split}] label 第 {line_no} 行非法: {lbl!r}"
                )
            labels[sid] = lbl
    if not labels:
        raise ValueError(f"[director:{split}] s1s2 label 文件为空: {s1s2_path}")
    print(f"[director:{split}] 加载 s1s2 labels: {len(labels)} 条")
    return labels


def run_director_stage(
    split: str,
    config: PipelineConfig,
    writer_records: Sequence[Dict[str, Any]],
    caller: "APICaller",
    limiter: RateLimiter,
) -> List[Dict[str, Any]]:
    paths = build_output_paths(config, split)
    existing_records = load_jsonl(paths["director"])
    completed_ids = {record["sample_id"] for record in existing_records}
    writer_items = [record for record in writer_records if record["split"] == split]
    # 加载 s1s2 分流 label；director 不允许在缺标签时回退为 S2。
    s1s2_labels = _load_s1s2_labels(config.output_dir, split)
    missing_label_ids = [
        record["sample_id"]
        for record in writer_items
        if record["sample_id"] not in s1s2_labels
    ]
    if missing_label_ids:
        preview = ", ".join(missing_label_ids[:5])
        raise ValueError(
            f"[director:{split}] {len(missing_label_ids)} 条 writer 样本缺少 S1/S2 label，"
            f"拒绝启动。示例: {preview}"
        )
    tasks = select_tasks(
        writer_items,
        completed_ids,
        config.debug_limit,
        config.random_seed,
        f"{split}:director",
    )
    if not tasks:
        print(f"[director:{split}] 没有待处理任务")
        return existing_records
    if config.director_provider_workers <= 0:
        raise ValueError("director_provider_workers 必须大于 0")
    if config.director_provider_quiet_window < 0:
        raise ValueError("director_provider_quiet_window 不能为负数")

    print(
        f"[director:{split}] 总剧本 {len(writer_items)}，"
        f"已完成 {len(completed_ids)}，待执行 {len(tasks)}"
    )
    records_by_id = {record["sample_id"]: record for record in existing_records}
    session_completed = 0
    session_total_tokens = 0
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_cached_tokens = 0
    session_cache_write_tokens = 0
    session_cost = 0.0
    session_dropped = 0
    provider_specs = resolve_director_provider_specs(config)
    providers = [
        DirectorProviderState(
            name=name,
            model_name=model_name,
            api_key=config.api_key,
            api_base=api_base,
            rpm=config.director_provider_rpm,
        )
        for name, model_name, api_base in provider_specs
    ]
    task_queue: Queue[DirectorTask] = Queue()
    for seq, writer_record in enumerate(tasks):
        task_queue.put(
            DirectorTask(
                seq=seq,
                writer_record=writer_record,
                s1_mode=s1s2_labels[writer_record["sample_id"]] == "S1",
            )
        )
    result_queue: Queue[Tuple[str, Any]] = Queue()
    stop_event = threading.Event()
    total_workers = config.director_provider_workers * len(providers)
    print(
        f"[director:{split}] provider 出水口: "
        f"{', '.join(f'{p.name}@{p.api_base}={p.model_name}' for p in providers)}；"
        f"每口 workers={config.director_provider_workers}，"
        f"每口 rpm={config.director_provider_rpm}"
    )

    def run_director_task(
        provider: DirectorProviderState,
        task: DirectorTask,
        worker_name: str,
    ) -> None:
        try:
            result = process_director_record_with_provider(task, config, provider)
            provider.mark_success(
                (
                    int(result["metadata"].get("director_completion_tokens", 0)),
                    int(result["metadata"].get("director_prompt_tokens", 0)),
                    int(result["metadata"].get("director_total_tokens", 0)),
                    {
                        "cached_tokens": int(result["metadata"].get("director_cached_tokens", 0)),
                        "cache_write_tokens": int(result["metadata"].get("director_cache_write_tokens", 0)),
                        "reasoning_tokens": int(result["metadata"].get("director_reasoning_tokens", 0)),
                    },
                )
            )
            result_queue.put(("success", result))
        except Exception as exc:
            if _is_drop_record_error(exc):
                provider.mark_error("task_error")
                result_queue.put(
                    (
                        "drop",
                        {
                            "sample_id": task.writer_record["sample_id"],
                            "provider": provider.name,
                            "worker": worker_name,
                            "attempts": task.attempts,
                            "error": str(exc),
                        },
                    )
                )
            elif _is_rate_limit_error(exc):
                provider.mark_429()
                task.attempts += 1
                if task.attempts <= config.max_retries:
                    task_queue.put(task)
                    result_queue.put(
                        (
                            "requeued",
                            {
                                "sample_id": task.writer_record["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc)[:500],
                            },
                        )
                    )
                else:
                    result_queue.put(
                        (
                            "failure",
                            {
                                "sample_id": task.writer_record["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc),
                            },
                        )
                    )
            elif _is_provider_fatal_error(exc):
                provider.disable(f"{type(exc).__name__}: {exc}")
                task_queue.put(task)
                result_queue.put(
                    (
                        "provider_disabled",
                        {
                            "sample_id": task.writer_record["sample_id"],
                            "provider": provider.name,
                            "worker": worker_name,
                            "error": str(exc)[:500],
                        },
                    )
                )
            else:
                task.attempts += 1
                provider.mark_error("task_error")
                if task.attempts <= config.max_retries:
                    task_queue.put(task)
                    result_queue.put(
                        (
                            "retry",
                            {
                                "sample_id": task.writer_record["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc)[:500],
                            },
                        )
                    )
                else:
                    result_queue.put(
                        (
                            "failure",
                            {
                                "sample_id": task.writer_record["sample_id"],
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": task.attempts,
                                "error": str(exc),
                            },
                        )
                    )

    def director_provider_worker(provider: DirectorProviderState, worker_idx: int) -> None:
        worker_name = f"{provider.name}-{worker_idx}"
        while not stop_event.is_set() and not provider.is_disabled():
            provider.limiter.wait_for_slot()
            try:
                task = task_queue.get(timeout=0.5)
            except Empty:
                continue
            if provider.is_disabled():
                task_queue.put(task)
                return
            run_director_task(provider, task, worker_name)

    def director_provider_dispatcher(provider: DirectorProviderState) -> None:
        with ThreadPoolExecutor(max_workers=config.director_provider_workers) as executor:
            futures = [
                executor.submit(director_provider_worker, provider, worker_idx)
                for worker_idx in range(config.director_provider_workers)
            ]
            for future in as_completed(futures):
                if stop_event.is_set() or provider.is_disabled():
                    break
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - worker guard
                    result_queue.put(
                        (
                            "worker_error",
                            {
                                "provider": provider.name,
                                "error": str(exc)[:500],
                            },
                        )
                    )

    threads: List[threading.Thread] = []
    for provider in providers:
        thread = threading.Thread(
            target=director_provider_dispatcher,
            args=(provider,),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    failures: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(tasks), desc=f"Director-{split}", smoothing=0)
    while session_completed + session_dropped + len(failures) < len(tasks):
        if all(provider.is_disabled() for provider in providers):
            stop_event.set()
            raise RuntimeError(f"[director:{split}] 所有 provider 都已禁用，无法继续")
        try:
            event_type, payload = result_queue.get(timeout=5.0)
        except Empty:
            if task_queue.empty() and not any(thread.is_alive() for thread in threads):
                break
            continue
        if event_type == "success":
            result = payload
            records_by_id[result["sample_id"]] = result
            meta = result.get("metadata", {})
            session_completed += 1
            p  = int(meta.get("director_prompt_tokens", 0))
            c  = int(meta.get("director_completion_tokens", 0))
            ca = int(meta.get("director_cached_tokens", 0))
            cw = int(meta.get("director_cache_write_tokens", 0))
            session_prompt_tokens     += p
            session_completion_tokens += c
            session_cached_tokens     += ca
            session_cache_write_tokens += cw
            session_total_tokens      += p + c
            model_name = str(meta.get("director_model_name") or config.model_name)
            session_cost              += _calc_cost(p, c, ca, model_name)
            avg_tokens = session_total_tokens / session_completed if session_completed else 0
            pbar.set_postfix({
                "Suc": session_completed,
                "AvgTok": int(avg_tokens),
                "Cost¥": f"{session_cost:.3f}",
                "Cache%": f"{session_cached_tokens/max(session_prompt_tokens,1)*100:.0f}%",
            })
            pbar.update(1)

            if session_completed % config.save_interval == 0:
                save_jsonl(
                    [records_by_id[key] for key in sorted(records_by_id)],
                    paths["director"],
                )
        elif event_type == "failure":
            failures.append(payload)
            pbar.update(1)
            tqdm.write(
                f"[director:{split}] {payload['sample_id']} 失败: "
                f"{payload['provider']} {payload['error']}"
            )
        elif event_type == "drop":
            session_dropped += 1
            pbar.update(1)
            tqdm.write(
                f"[director:{split}] {payload['sample_id']} drop: "
                f"{payload['provider']} {payload['error']}"
            )
        elif event_type == "requeued":
            pbar.set_postfix({"429": payload["provider"], "Suc": session_completed})
        elif event_type == "provider_disabled":
            pbar.set_postfix({"Disable": payload["provider"], "Suc": session_completed})
            tqdm.write(
                f"[director:{split}] provider {payload['provider']} 已禁用: "
                f"{payload['error']}"
            )
        elif event_type == "retry":
            pbar.set_postfix({"Retry": payload["provider"], "Suc": session_completed})
        elif event_type == "worker_error":
            pbar.set_postfix({"WorkerErr": payload["provider"], "Suc": session_completed})
            tqdm.write(
                f"[director:{split}] provider {payload['provider']} worker 异常: "
                f"{payload['error']}"
            )

    stop_event.set()
    for thread in threads:
        thread.join(timeout=2.0)
    for provider in providers:
        provider.close()
    pbar.close()
    incomplete = len(tasks) - session_completed - session_dropped - len(failures)
    if incomplete:
        raise RuntimeError(
            f"[director:{split}] {incomplete} 条 director 任务未完成，拒绝静默完成"
        )

    final_records = [records_by_id[key] for key in sorted(records_by_id)]
    save_jsonl(final_records, paths["director"])
    print(f"[director:{split}] 完成后总记录数: {len(final_records)}")
    if session_dropped:
        print(f"[director:{split}] 本轮 400/drop 样本数: {session_dropped}")
    if failures:
        raise RuntimeError(f"[director:{split}] {len(failures)} 条 director 任务失败，已拒绝静默完成")
    if session_completed:
        print(
            f"[director:{split}] Token 统计 — "
            f"prompt={session_prompt_tokens:,}  "
            f"completion={session_completion_tokens:,}  "
            f"cached={session_cached_tokens:,}("
            f"{session_cached_tokens/max(session_prompt_tokens,1)*100:.1f}%)  "
            f"cache_write={session_cache_write_tokens:,}  "
            f"cost=¥{session_cost:.4f}"
        )
        for provider in providers:
            snap = provider.snapshot()
            stats = snap["stats"]
            prompt = int(stats.get("prompt_tokens", 0))
            cached = int(stats.get("cached_tokens", 0))
            cost = _calc_cost(
                prompt,
                int(stats.get("completion_tokens", 0)),
                cached,
                snap["model_name"],
            )
            print(
                f"[director:{split}] provider {snap['name']} — "
                f"api_base={snap['api_base']} "
                f"model={snap['model_name']} "
                f"success={stats.get('success', 0)} "
                f"429={stats.get('429', 0)} "
                f"task_error={stats.get('task_error', 0)} "
                f"paused={stats.get('worker_paused', 0)} "
                f"cache={cached/max(prompt,1)*100:.1f}% "
                f"cost=¥{cost:.4f}"
            )
    return final_records


def resolve_splits(config: PipelineConfig) -> List[str]:
    if config.split == "both":
        return ["inbound", "outbound"]
    return [config.split]


def main() -> None:
    config = parse_args()
    os.makedirs(config.output_dir, exist_ok=True)

    caller = create_api_caller(config)
    limiter = RateLimiter(config.rpm_limit)

    all_seed_items: List[Dict[str, Any]] = []
    if config.mode != "director_only":
        all_seed_items = load_seed_items(config.seed_path)
        if len(all_seed_items) < config.writer_seed_sample_size:
            raise ValueError("seed 数量不足，无法启动编剧阶段")

    split_to_path = {
        "inbound": config.inbound_path,
        "outbound": config.outbound_path,
    }

    for split in resolve_splits(config):
        print(f"\n========== 处理 split: {split} ==========")
        content_items = load_content_items(split_to_path[split], split)
        if not content_items:
            print(f"[{split}] 未读取到有效 content")
            continue

        if config.mode == "writer_only":
            run_writer_stage_with_provider_pool(
                split,
                config,
                content_items,
                all_seed_items,
            )
            continue

        if config.mode == "director_only":
            writer_path = Path(config.output_dir) / f"{split}.writer.jsonl"
            writer_records = load_jsonl(str(writer_path))
            if not writer_records:
                print(f"[director:{split}] 未找到编剧中间文件: {writer_path}")
                continue
            run_director_stage(split, config, writer_records, caller, limiter)
            continue

        writer_records = run_writer_stage_with_provider_pool(
            split,
            config,
            content_items,
            all_seed_items,
        )
        run_director_stage(split, config, writer_records, caller, limiter)


if __name__ == "__main__":
    main()
