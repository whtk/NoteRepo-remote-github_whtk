> 来自课程：[DLAI - Learning Platform Prototype (deeplearning.ai)](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)


prompt 原则：
1. clear and specific prompt（clear 不等于 short）
2. give the model time to think

原则 1 技巧：
+ 使用分隔符清楚地指示输入的不同部分
	+ 如：概括 {} 内的句子（而不是说 概括以下句子）
+ 要求结构化的输出：
	+ 如，要求模型输出 html 或者 json 格式的输出
+ 要求模型确认要求是否被满足
+ few shot prompting

原则 2 技巧：
+ 手动给出完成任务的步骤
+ 指导模型在匆忙得出结论之前制定自己的解决方案

如何避免模型“一本正经的胡说八道”:
+ 要求模型找出相关的信息
+ 基于相关信息找出问题的答案

迭代式 prompt 原则：
+ 提出 idea
+ 写出 prompt
+ 根据结果 refine idea
+ 修改 prompt
+ 重复以上步骤

ChatGPT 用于文本总结：
+ 可以给出总结的重点，如，要求模型关注某一部分
+ 考虑将 prompt 改为 “提取” 而非 “总结”

ChatGPT 用于推理：
+ 判断文本的情感色彩（正向 or 负向 ）
+ 情感分类

ChatGPT 用于翻译（或转换）：
+ 用处：翻译、格式转换（如 json 转 html）等
+ 语气转换
+ 语法检查

ChatGPT 用于拓展（将短文本拓展为长文本）：
+ 注意调整 temperature

ChatGPT 用于聊天：
+ 角色扮演
+  用于各种助理
