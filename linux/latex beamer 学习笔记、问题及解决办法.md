
## Beamer 学习笔记

> 目前只适用于纯英文的模版，译自 [Overleaf-Beamer](https://www.overleaf.com/learn/latex/Beamer)。

### 最基础的通用模版（纯英文）

```latex
\documentclass{beamer}
\usetheme{Boadilla} % 可以写不同的 theme

\title{My Presentation}
\subtitle{Using Beamer}
\author{Joe Bloggs}
\institute{University of ShareLaTeX}
\date{\today}

\begin{document}

	\begin{frame}
	\titlepage
	\end{frame}

\end{document}
```
<!-- The font size, here 17pt, can be passed as a parameter to the beamer class at the beginning of the document preamble: \documentclass[17pt]{beamer}. Below is an example showing the result of using the 17pt font-size option: -->
> 可以在文档的开头将字体大小（例如 17pt）作为参数传递给 beamer 类：`\documentclass[17pt]{beamer}`。

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

### 如何添加按钮（略）

### 关于 title 页

以 Boadilla 模版为例：

`\title[]{}` 中，`[]` 内容显示在整个模版的底部中间（所谓的短标题），`{}` 为正式的标题。

`\author[]{}` 中，`[]` 内容显示在整个模版的底部左边，`{}` 为正式的作者。

`\institute[]{}` 中，`[]` 内容显示在整个模版的底部左边作者旁边（一般是机构/学校的简写，显示的时候会加个括号），`{}` 为完整的机构。

`\date[]{}` 中，`[]` 内容显示在整个模版的底部右边，`{}` 为正式的日期。

使用例子如下：

```latex
\title[About Beamer] %optional
{About the Beamer class in presentation making}

\subtitle{A short story}

\author[Arthur, Doe] % (optional, for multiple authors)
{A.~B.~Arthur\inst{1} \and J.~Doe\inst{2}}

\institute[VFU] % (optional)
{
  \inst{1}%
  Faculty of Physics\\
  Very Famous University
  \and
  \inst{2}%
  Faculty of Chemistry\\
  Very Famous University
}

\date[VLC 2021] % (optional)
{Very Large Conference, April 2021}

\logo{\includegraphics[height=1cm]{overleaf-logo}}
```

### 一些其他的 effect

#### pause 命令
<!-- Often when when doing a presentation we'll want to reveal parts of a frame one after the other. The simplest way to do this is to use the \pause command. For example, by entering the \pause command before every entry in a list we can reveal the list point-by-point: -->
通常在做演示时，我们会想要逐步显示幻灯片的部分内容。最简单的方法是使用 `\pause` 命令。例如，在列表中的每个条目之前输入 `\pause` 命令，可以逐点显示列表：
    
```latex
\begin{frame}
    \frametitle{List}
    \begin{itemize}
        \pause
        \item Point A

        \pause
        \item Point B

        \begin{itemize}
            \pause
            \item part 1

            \pause
            \item part 2
        \end{itemize}

        \pause
        \item Point C
        \pause
        \item Point D
    \end{itemize}
\end{frame}
```

#### Overlays
<!-- The \pause command is useful but isn't very versatile. To get more flexibility we use what beamer calls overlay specifications. These specifications can be added to compatible commands using pointed brackets after the command name. For example I can add them to the \item command in a list structure like this. -->
`pause` 命令很有用，但不够灵活。为了获得更多的灵活性，可以使用 overlay。可以在命令名后的尖括号中添加到兼容的命令中。例如，我可以将它们添加到列表结构中的 `item` 命令中：

```latex
\begin{frame}
    \frametitle{More Lists}
    \begin{enumerate}[(I)]
        \item<1-> Point A
        \item<2-> Point B
            \begin{itemize}
            \item<3-> part 1
            \item<4-> part 2
            \end{itemize}
        \item<5-> Point C
        \item<6-> Point D
    \end{enumerate}
\end{frame}
```

<!-- The numbers inside the pointed brackets tell LaTeX which slides the item should appear on. For example, in this list we've told each list item which slide number it should first appear on and then told them to appear on all subsequent slides in the frame using the dash. Here's an example of a more complicated overlay: -->
尖括号中的数字告诉 LaTeX 项目应该出现在哪些幻灯片上。例如，在这个列表中，我们告诉每个列表项它应该首先出现在哪个幻灯片上，然后使用破折号告诉它们在幻灯片中的所有后续幻灯片上出现。下面是一个更复杂的 overlay 的例子：

```latex
\item<-2,4-5,7>
```
<!-- This makes the item appear on slides 1,2,4,5 & 7. -->
此命令使项目出现在幻灯片 1、2、4、5 和 7 上。

<!-- There are a number of commands that enable us to use overlays on text. The main one is the \onslide command which can be configured to achieve a few different outcomes, details of these can be found in the documentation. -->
有一些命令可以让我们在文本上使用 overlay。主要的一个是 `\onslide` 命令，可以配置为实现一些不同的结果：

```latex
\begin{frame}
\frametitle{Overlays}
\onslide<1->{First Line of Text}

\onslide<2->{Second Line of Text}

\onslide<3->{Third Line of Text}
\end{frame}
```

<!-- To make the text transparent on unspecified slides we use the \setbeamercovered command and enter the keyword transparent above the code where we want it to have an effect: -->
为了使文本在未指定的幻灯片上透明，我们使用 `\setbeamercovered` 命令，并在我们想要它生效的代码上方输入关键字 `transparent`：

```latex
\setbeamercovered{transparent}
```

<!-- Please be aware that this command will affect all of the code following it, so if we want to change it back to the default setting later in the presentation we can simply use the same command again but with the keyword invisible. -->
请注意，这个命令将影响其后的所有代码，所以如果我们想在演示的后面将其改回默认设置，我们可以简单地再次使用相同的命令，但关键字为 `invisible`。
<!-- The \invisble command does the exact opposite to of the \visible command. The \only command does the same as the \visible command except it doesn't take any space up. This means that if we change the \onslide commands to \only commands and get rid of the dashes in the overlay specifications our three lines of text will appear in the same place on the frame in turn. -->
`invisible` 命令与 `visible` 命令完全相反。`only` 命令与 `visible` 命令相同，只是它不占用任何空间。这意味着如果我们将 `onslide` 命令更改为 `only` 命令，并且去掉 overlay 规范中的破折号，下面三行文本将依次出现在幻灯片的同一位置：
    
```latex
\begin{frame}
    \frametitle{Overlays}
    \only<1>{First Line of Text}

    \only<2>{Second Line of Text}

    \only<3>{Third Line of Text}
\end{frame}
```

#### 主题

不同的主题有不同的样式，可以在 `\usetheme{}` 中选择。

从组成来看，theme 可以分为：
+ color theme
+ font theme
+ inner theme
+ outer theme

且可以分别设置。例如，采用 `\usecolortheme` 设置颜色主题，`\usefonttheme` 设置字体主题，`\useinnertheme` 设置内部主题，`\useoutertheme` 设置外部主题。
> 注意这个命令是放在 `\usetheme{}` 之后。
<!-- The \usefonttheme{} is self-descriptive. The available themes are: structurebold, structurebolditalic, structuresmallcapsserif, structureitalicsserif, serif and default. -->
> `\usefonttheme{}` 是 self-descriptive 的。可用的主题有：structurebold、structurebolditalic、structuresmallcapsserif、structureitalicsserif、serif 和 default。也可以导入其他字体包。
<!-- The inner theme dictates the style of the title and part pages, the itemize, enumerate, description, block, theorem and proof environments as well as figures, tables, footnotes and bibliography entries. For example we could also load up the rectangles inner theme. We do this using the \useinnertheme command. This has made our table of contents and lists use rectangles as bullet points: -->
inner theme 决定了标题和部分页面的样式，itemize、enumerate、description、block、theorem 和 proof 环境，以及图、表、脚注和参考文献条目的样式。例如，我们还可以加载矩形内部主题。我们可以使用 `\useinnertheme` 命令来实现。这使得我们的目录和列表使用矩形作为项目符号：
<!-- The outer theme dictates the style of the head and footline, the logo, the sidebars and the frame title. We can specify this theme using the \useoutertheme command. As we're using Warsaw, by default we are using the shadow outer theme, but we could change this to the tree theme if we wanted to change the top navigation bar to a tree like structure: -->
outer theme 决定了页眉和页脚、标志、侧边栏和帧标题的样式。我们可以使用 `\useoutertheme` 命令指定这个主题。如果使用的是 Warsaw，默认情况下用的是 shadow 外部主题，但是如果想要将顶部导航栏更改为树状结构，我们可以将其更改为 tree 主题。

#### handouts
<!-- Now let's briefly look at creating handouts for our presentation. To do this we add the keyword handout into square brackets in the document class command. We then use the pgfpages package to help us print multiple slides on a page. After loading the package we use the \pgfpagesuselayout command. In the curly brackets we specify how many frames we want on a sheet. In the square brackets we specify the paper size and how much border shrink we want: -->
在在文档类命令中的方括号中添加关键字 `handout`。然后使用 `pgfpages` 包来在一页上打印多个幻灯片。加载包后，使用 `\pgfpagesuselayout` 命令。在大括号中，指定我们想要在一张纸上有多少幻灯片。在方括号中，指定纸张大小和边框收缩：

```latex
\documentclass[handout]{beamer}
\usepackage{pgfpages}
\pgfpagesuselayout{2 on 1}[a4paper,border shrink=5mm]
```
<!-- If we wanted to put four frames on a sheet we could simply change the 2 to a 4 and then add the landscape keyword into the square brackets: -->
如果我们想在一张纸上放四个幻灯片，我们可以简单地将 2 改为 4，然后在方括号中添加 landscape 关键字：

```latex
\pgfpagesuselayout{4 on 1}[a4paper,border shrink=5mm,landscape]
```



## 问题及其解决办法

问题：Package rerunfilecheck: File .out has changed.  (rerunfilecheck) Rerun to get outlines right  (rerunfilecheck) or use package `bookmark'.
解决办法：\usepackage{bookmark}

## 学习资源和参考资料

1. [Overleaf-Beamer 官网](https://www.overleaf.com/learn/latex/Beamer)
2. [Beamer Class User Guide for version 3.71.](http://texdoc.net/pkg/beamer)
3. [中译版 Beamer 用户指南 3.24.](https://www.latexstudio.net/archives/9457.html)
4. 官方 beamer 模版，可以在 [github](https://github.com/josephwright/beamer/tree/main/doc) 上找到，或者在本地位置 /usr/local/texlive/2023/texmf-dist/doc/latex/beamer 找到（Mac 系统）；对于特定场景的模版（如 20 min 会议模版），位于 beamer/solutions/conference-talks 位置下
5. [不同模版可视化](https://mpetroff.net/files/beamer-theme-matrix/)