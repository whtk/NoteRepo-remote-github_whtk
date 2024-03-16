
## 笔记

### 最基础的通用模版（纯英文）

```latex
\documentclass{beamer}
\usetheme{Boadilla} % 可以写不同的 theme

\begin{document}

	\begin{frame}
	\titlepage
	\end{frame}

\end{document}
```

### 如何添加 logo

```latex
\logo{\includegraphics[width=xxx]{Logo.png}} % 在 \begin{document} 之前，{} 内也可以是文字
```
注意：logo 的位置取决于主题，如果只想在 title 页面显示 logo，可以用 `\titlegraphic{}` 而非 `\logo{}`。

### 如何添加 outline 
<!-- The \tableofcontents command creates the table of contents as it did in LaTeX. The table automatically gets updated with the addition or removal of sections and subsections. We have to create a frame environment and we add the command in question. -->
    
用 `\tableofcontents` 命令创建目录。当添加或删除章节和子章节时目录会自动更新。用法如下：

```latex
\begin{frame}{Outline}
    \tableofcontents % \tableofcontents[hideallsubsections] 可以隐藏子章节
\end{frame}
```

<!-- It is also possible to create a recurring table of contents before every section. This highlights the current section and fades out the rest. This feature is used to remind the audience of where we are in the presentation. This can be done with the help of \AtBeginSection command and specifying [currentsection] in the \tableofcontents command. Please go through the example below for better understanding: -->
也可以在每个章节前创建一个重复的目录。这样可以突出当前章节并淡化其他章节。这个功能可以用 `\AtBeginSection` 命令和在 `\tableofcontents` 命令中指定 `[currentsection]` 来实现。示例：

```latex
\AtBeginSection[]
{
\begin{frame}{Outline}
    \tableofcontents[currentsection]
\end{frame}
}
```

### 如何创建列表
<!-- Lists can be created using three environments in beamer: \enumerate, \itemize, and \description. Nested lists can also be created by combining these environments. To create an entry in these environments, the \item command is used.
 -->
列表可以用三个环境创建：`\enumerate`、`\itemize` 和 `\description`。用 `\item` 创建列表项。其中，
<!-- Itemize is used to create unordered lists. Under this environment, the obtained list will have bullet points. Check the following code: -->
`\itemize` 用于创建无序列表，列表项前会有圆点。
<!-- There are various templates in beamer to change this itemized list appearance. The command \setbeamertemplate is used on itemize items to change the shape of item markers. -->
> 可以通过 `\setbeamertemplate` 命令改变列表项的形状。
> `\setbeamertemplate{itemize items}[default]` 为默认形状，`ball` 为圆形，`triangle` 为三角形，`circle` 为圆圈，`square` 为正方形

<!-- This environment is used to create an ordered list. By default, before each item increasing Arabic numbers followed by a dot are printed (eg. “1.” and “2.”). -->
`\enumerate` 用于创建有序列表，默认情况下，每个列表项前会有递增的阿拉伯数字和点号（例如“1.” 和 “2.”）。
> 同样可以通过 `\setbeamertemplate{enumerate items}[circle]` 命令改变列表项的形状。

<!-- The description environment is used to define terms or to explain acronyms. We provide terms as an argument to the \item command using squared bracket. -->
`\description` 用于定义术语或解释缩写。用方括号给 `\item` 命令提供术语。如：
    
```latex
\begin{description}
    \item[Term 1] Explanation 1
    \item[Term 2] Explanation 2
\end{description}
```

### 如何创建图表
图表的创建和 LaTeX 中一样，例子如下：
    
```latex
    % Tables in beamer
\begin{frame}{Simple table in beamer}
    \begin{table}
        \begin{tabular}{| c | c | c |}
            \hline
            No. & Name & Age \\
            \hline \hline
            1 & John T & 24 \\
            2 & Norman P & 8 \\
            3 & Alex K & 14 \\ 
            \hline
        \end{tabular}
        \caption{Name and age of students}
    \end{table}

    \begin{figure}
        \includegraphics[scale=0.5]{xx.png}
        \caption{XXX}
    \end{figure}
\end{frame}
```

### 如何创建列（column）
<!-- Columns can be created in beamer using the environment named columns. Inside this environment, you can either place several column environments, each of which creates a new column, or use the \column command to create new columns. -->
列可以用 `columns` 环境创建。在这个环境中，可以放置多个 `column` 环境，每个 `column` 环境创建一个新的列，也可以用 `\column` 命令创建新的列。例子如下：

```latex
% Multicolumn frame in beamer
\begin{frame}{Two columns frame in beamer}

    \begin{columns}
        % Column 1
        \begin{column}{0.5\textwidth}
            Text here! Text here! ...
        \end{column}

        % Column 2
        \begin{column}{0.5\textwidth}
            \includegraphics[scale=0.5]{Beamer-Logo.png}
        \end{column}
    \end{columns}

\end{frame}
```

### 如何创建块（block）
<!-- Information can be displayed in the form of blocks using block environment. These blocks can be of three types :

alert block.
example block.
and theorem block. -->
可以用 `block` 环境创建块。这些块可以是三种类型：警告块、示例块和定理块。
<!-- The standard block is used for general text in presentations. It has a blue color and can be created as follows: -->
标准块用于演示中的一般文本。它是蓝色的，可以用如下方式创建：

```latex
% Blocks in beamer
\begin{frame}{Blocks in beamer}{}
    \begin{block}{Block 1}
        This is a simple block in beamer.
    \end{block}
\end{frame}
```
<!-- The purpose of the alert block is to stand out and draw attention towards the content. This block is used to display warning or prohibitions. The default color of this block is red. To display an alert block the code can be written as: -->
警告块的目的是突出显示并引起注意。这个块用于显示警告或禁止。这个块的默认颜色是红色。可以用如下方式创建：

```latex
% Blocks in beamer
\begin{frame}{Blocks in beamer}{}
    \begin{alertblock}{Block 2}
        This is an alert block in beamer.
    \end{alertblock}
\end{frame}
```
<!-- This block is used to highlight examples as the name suggests and it can also be used to highlight definitions. The default color of this block is green and it can be created as follows: -->
示例块用于突出显示示例，它的默认颜色是绿色，可以用如下方式创建：

```latex
% Blocks in beamer
\begin{frame}{Blocks in beamer}{}
    \begin{exampleblock}{Block 3}
        This is an example block in beamer.
    \end{exampleblock}
\end{frame}
```
<!-- The theorem block is used to display mathematical equations, theorems, corollary and proofs. The color of this block is blue. Here is an example: -->
定理块用于显示数学公式、定理、推论和证明。这个块的颜色是蓝色。例子如下：

```latex
% Blocks in beamer
\begin{frame}{Math related blocks in Beamer}{Theorem, Corollary and Proof}

    \begin{theorem}
        It's in \LaTeX{} so it must be true $ a^2 + b^2 = c^2$.
    \end{theorem}

    \begin{corollary}
        a = b
    \end{corollary}

    \begin{proof}
        a + b = b + c
    \end{proof}

\end{frame}
```

### 关于 title 页

`\title[]{}` 中，`[]` 内容显示在整个模版的底部中间（所谓的短标题），`{}` 为正式的标题。

`\author[]{}` 中，`[]` 内容显示在整个模版的底部左边，`{}` 为正式的作者。

`\institute[]{}` 中，`[]` 内容显示在整个模版的底部左边作者旁边，`{}` 为正式的机构。

`\date[]{}` 中，`[]` 内容显示在整个模版的底部右边，`{}` 为正式的日期。

### 一些其他的 effect

#### 列表逐步显示
<!-- In the introduction, we saw a simple slide using the \begin{frame} \end{frame} delimiters. It was mentioned that a frame is not equivalent to a slide, and the next example will illustrate why, by adding some effects to the slideshow. In this example, the PDF file produced will contain 4 slides—this is intended to provide a visual effect in the presentation. -->
下面的例子中，生成的 PDF 文件将包含 4 个幻灯片，可以在演示中提供视觉效果：

```latex
\begin{frame}
\frametitle{Sample frame title}
This is a text in second frame. 
For the sake of showing an example.

\begin{itemize}
    \item<1-> Text visible on slide 1 
    \item<2-> Text visible on slide 2
    \item<3> Text visible on slide 3
    \item<4-> Text visible on slide 4
\end{itemize}
\end{frame}
```


## 问题及其解决办法

Package rerunfilecheck: File .out has changed.  (rerunfilecheck) Rerun to get outlines right  (rerunfilecheck) or use package `bookmark'.
> \usepackage{bookmark}

