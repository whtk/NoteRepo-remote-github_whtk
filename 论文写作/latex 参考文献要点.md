
标点在参考文献之后。

引用参考文献时，直接写 as in 【1】，不用写 Ref. 【1】，除非是在具体的开头写：Reference 【1】was the first..

除非有6个及以上的作者名单，否则不要使用 et al.

引用未发表的文章，应该引用为 unpublished；已经被接收发表的文章引用为 in press。除了专有名词和元素符号外，只将论文标题中的第一个单词大写。

如何修改 IEEEtran.bst 模版中参考文献的字体和行间距：
+ 行间距：修改 `\def\IEEEbibitemsep{0pt plus 1.0pt}` ，其中  `0pt` 指定参考文献条目之间的**最小间距**为 0pt（即零长度）；`5pt` 指定参考文献条目之间的**可拉伸间距**为 0.5pt，这意味着如果页面排版需要调整以避免不美观的分布，LaTeX 可以将条目间距拉伸到最多 0.5pt 以优化页面布局。
+ 字体大小：找到下面的代码：
```latex
\def\thebibliography#1{\section*{\refname}%
    \addcontentsline{toc}{section}{\refname}%
    % V1.6 add some rubber space here and provide a command trigger
    \footnotesize\vskip 0.3\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip%
    \list{\@biblabel{\@arabic\c@enumiv}}%
    {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep\relax
    \itemsep \IEEEbibitemsep\relax
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    \let\@IEEElatexbibitem\bibitem%
    \def\bibitem{\@IEEEbibitemprefix\@IEEElatexbibitem}%
\def\newblock{\hskip .11em plus .33em minus .07em}%
```
将其中的 `footnotesize` 改成目标字体即可