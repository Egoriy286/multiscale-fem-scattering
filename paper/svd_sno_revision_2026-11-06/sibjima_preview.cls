\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sibjim}[2019/06/26.  Version 1.  Nechaev~D.]
\RequirePackage{amsmath,amssymb,amstext,bm}
\LoadClass[11pt,twoside]{article}
\RequirePackage[russian]{babel}
\RequirePackage{graphicx}
\RequirePackage{floatflt}

\let\oldcaption\caption

\newif\if@titled \@titledtrue
\DeclareOption{notitle}{\@titledfalse}
\ProcessOptions

\ifx\outformat\undefined \newcount\outformat\outformat=0\fi

\ifcase\outformat
\or
\pdfpagewidth=205mm
\pdfpageheight=290mm
\fi

\topmargin -2mm
\oddsidemargin -3mm
\evensidemargin -3mm
\textwidth 165mm
\textheight 235mm
\topsep 1.3ex plus0.2ex minus1ex
\parindent 8mm
\partopsep 0pt
\marginparwidth 85pt
\marginparsep 10pt
\headheight 14pt
\headsep 15pt

\overfullrule=5pt

\DeclareMathSizes{\@xiipt}{\@xiipt}{9}{8}

\def\set#1#2#3{\expandafter\def\csname jim@#1@#2\endcsname{#3}}
\def\get#1#2{\csname jim@#1@#2\endcsname}

\def\issue#1#2#3{\set{paper}{year}{#1}\set{paper}{tom}{#2}\set{paper}{number}{#3}}

\def\title{\@ifnextchar[{\title@short@full}{\title@full}}
\def\title@short@full[#1]#2{\set{title}{short}{#1}\set{title}{full}{#2}}
\def\title@full#1{\set{title}{short}{#1}\set{title}{full}{#1}}

\def\udk#1{\set{paper}{udk}{#1}}

\def\doi#1{\set{paper}{doi}{#1}}
\set{paper}{doi}{\relax}

\def\workplace#1{\set{paper}{workplace}{#1}}

\def\email#1#2{\set{email}{#1}{#2}}

\def\pages#1#2{\set{page}{first}{#1}\set{page}{last}{#2}\setcounter{page}{#1}}

\def\authors{\@ifnextchar[{\authors@short@full}{\authors@full}}
\def\authors@short@full[#1]#2{\set{authors}{short}{#1}\set{authors}{full}{#2}}
\def\authors@full#1{\set{authors}{short}{#1}\set{authors}{full}{#1}}

\def\abstract#1{\set{paper}{abstract}{#1}}

\def\keys#1{\set{paper}{keys}{#1}}

\def\dates#1#2#3{\set{date}{received}{#1}\set{date}{revised}{#2}%
\set{date}{published}{#3}}

\def\grant#1{\set{paper}{grant}{\null#1}}

\def\author#1#2{\set{author}{#1}{\null#2}}

\def\maketitle{
\noindent{\footnotesize УДК \get{paper}{udk}}

\par\vspace{10pt}
\centerline{\parbox{0.9\textwidth}{\centering\bf\get{title}{full}%
%\if\get{paper}{grant}\relax\else $^{*}$\fi
}}

\par\vspace{20pt}
\centerline{\parbox{0.9\textwidth}{\centering\bf\copyright\;\get{paper}{year}\q
\get{authors}{full}}}

\par\vspace{10pt}
\centerline{\parbox{0.9\textwidth}{\centering\it\get{paper}{workplace}}}

\if\get{email}{a}\relax\else
\par\vspace{20pt}
\centerline{\parbox{0.9\textwidth}{\centering
\if\get{email}{b}\relax E-mail\else E-mails\fi:\q
\if\get{email}{b}\relax\else $^a$\fi\get{email}{a}%
\if\get{email}{b}\relax\else{\if\get{email}{a}\relax\else,\q\fi}$^b$\get{email}{b}\fi
\if\get{email}{c}\relax\else{\if\get{email}{b}\relax\else,\q\fi}$^c$\get{email}{c}\fi
\if\get{email}{d}\relax\else{\if\get{email}{c}\relax\else,\q\fi}$^d$\get{email}{d}\fi
\if\get{email}{e}\relax\else{\if\get{email}{d}\relax\else,\q\fi}$^e$\get{email}{e}\fi
\if\get{email}{f}\relax\else{\if\get{email}{e}\relax\else,\q\fi}$^f$\get{email}{f}\fi
\if\get{email}{g}\relax\else{\if\get{email}{f}\relax\else,\q\fi}$^g$\get{email}{g}\fi
\if\get{email}{h}\relax\else{\if\get{email}{g}\relax\else,\q\fi}$^h$\get{email}{h}\fi
\if\get{email}{i}\relax\else{\if\get{email}{h}\relax\else,\q\fi}$^i$\get{email}{i}\fi
}}\fi

\par\vspace{10pt}
\centerline{\parbox{0.9\textwidth}{\centering\small
Поступила в редакцию~\get{date}{received}~г.;
после доработки~\get{date}{revised}~г.;\\
принята к публикации~\get{date}{published}~г.}}
\par\vspace{20pt}

\centerline{\parbox{145mm}{\fontsize{10}{11}\selectfont
\get{paper}{abstract}}}
\par\vspace{10pt}

\centerline{\parbox{145mm}{\fontsize{10}{11}\selectfont
{\bf Ключевые слова:} \get{paper}{keys}}}
\par

\ifx\get{paper}{doi}\undefined\else
\vspace{10pt}\centerline{\parbox{145mm}{\fontsize{10}{11}\selectfont
DOI: 10.33048/SIBJIM.\get{paper}{year}.\get{paper}{tom}.\get{paper}{number}\get{paper}{doi}}}\par\fi

%\if\get{paper}{grant}\relax\else\fnote{$^{*}\,$}{\get{paper}{grant}.}\fi
\if\get{paper}{grant}\relax\else\fnote{}{\get{paper}{grant}}\fi
}

\def\fnote#1#2{\def\thempfn{#1}\footnotetext{#2}\addtocounter{footnote}{-1}%
\let\thempfn=\thefootnote}

\def\footnoterule{\kern2\p@ \hrule width .2\columnwidth\kern 3\p@}

\def\ps@jim{
  \def\@oddhead{\vbox{\hbox to \textwidth{\footnotesize\hskip5mm\head@odd\hfill\thepage\strut}
  \hrule height 0.4pt width\textwidth}}
  \def\@evenhead{\vbox{\hbox to \textwidth{\footnotesize\thepage\hfill\head@even\hskip5mm\strut}
  \hrule height 0.4pt width\textwidth}}
  \def\@oddfoot{\empty}  \def\@evenfoot{\empty}}

\def\ps@title{
  \def\@oddhead{\vbox{\hbox to \textwidth{\footnotesize\hskip5mm \jim@head\hfill\strut}
  \hrule height 0.4pt width\textwidth}}
  \def\@evenhead{\vbox{\hbox to \textwidth{\footnotesize\hfill \jim@head\hskip5mm\strut}
  \hrule height 0.4pt width\textwidth}}
  \def\@oddfoot{\empty}  \def\@evenfoot{\empty}}

\def\jim@head{CИБИРСКИЙ ЖУРНАЛ ИНДУСТРИАЛЬНОЙ МАТЕМАТИКИ.
\get{paper}{year}. Т.~\get{paper}{tom}, №~\get{paper}{number}.
C.~\get{page}{first}--\get{page}{last}}


\def\head@odd{\get{title}{short}}
\def\head@even{\get{authors}{short}}


\pagestyle{jim}

\def\sbc{\fontfamily{cmss}\fontseries{sbc}\selectfont}

\def\auth#1{\textsl{#1}}

\sloppy

\AtBeginDocument{
\thispagestyle{title}
\if@titled\maketitle\vspace{10pt minus5pt}\fi
\baselineskip=13.4pt plus 0.2pt minus 0.1pt}

\AtEndDocument{\par\bigskip
\if\get{author}{1}\else \noindent\textit{\get{author}{1}}\fi
\if\get{author}{2}\else \par\noindent\textit{\get{author}{2}}\fi
\if\get{author}{3}\else \par\noindent\textit{\get{author}{3}}\fi
\if\get{author}{4}\else \par\noindent\textit{\get{author}{4}}\fi
\if\get{author}{5}\else \par\noindent\textit{\get{author}{5}}\fi
\if\get{author}{6}\else \par\noindent\textit{\get{author}{6}}\fi
\if\get{author}{7}\else \par\noindent\textit{\get{author}{4}}\fi
\if\get{author}{8}\else \par\noindent\textit{\get{author}{5}}\fi
\if\get{author}{9}\else \par\noindent\textit{\get{author}{6}}\fi
}

\def\thebibliography#1{
\kern5mm
\begin{center}
  {ЛИТЕРАТУРА}
\end{center}
\small
\list
 {\arabic{enumi}.}{\settowidth\labelwidth{#1.}\leftmargin\labelwidth
 \advance\leftmargin\labelsep
 \usecounter{enumi}\itemsep0.1pt}
 \def\newblock{\hskip .11em plus .33em minus .07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}

\def\section{\@ifnextchar*{\s@ctiona}{\s@ction}}

\def\s@ction#1{\refstepcounter{section}\par\medskip
\begin{center}\bf\thesection. #1\end{center}\smallskip\nopagebreak}

\def\s@ctiona#1#2{\par\medskip
\begin{center}\bf #2\end{center}\smallskip\nopagebreak}

\def\subsection{\@ifnextchar*{\subs@ctiona}{\subs@ction}}

\def\subs@ction#1{
\refstepcounter{subsection}
\begin{center}\bf\thesubsection. #1\end{center}
\nopagebreak}

\def\subs@ctiona#1#2{
\begin{center}\bf #2\end{center}
\nopagebreak}

\def\subsubsection{\@ifnextchar*{\subsubs@ctiona}{\subsubs@ction}}

\def\subsubs@ction#1{
\refstepcounter{subsubsection}
\textbf{\thesubsubsection. #1.}
\nopagebreak}

\def\subsubs@ctiona#1#2{
\textbf{#2.}
\nopagebreak}


\def\topfraction{1.0}
\def\bottomfraction{1.0}

\let\leq\le
\let\geq\ge
\let\qq\qquad
\let\q\quad
\let\ds\displaystyle
\let\ge\geqslant
\let\le\leqslant

\def\vniz#1#2{\raisebox{-#1em}[0pt][0pt]{#2}}
\def\vUP{\rule{0pt}{2.8ex}}
\def\tabox#1{\kern-0.5em{\renewcommand{\arraystretch}{1}\begin{tabular}{c} \\[-4pt] #1\\[2pt] \end{tabular}}\kern-0.7em}
\newcommand{\vshir}[2]{\hbox to #1em{\hss #2\hss}}

\newcommand\Figure[3]{\leavevmode\ifcase\outformat%
\hbox{\vrule\vbox to #2mm{\hrule\vfill\hbox to #1mm{\ #3\hfill}\vfill\hrule}\vrule}\or %dvi
\IfFileExists{#3.pdf}{\hbox{\pdfximage width #1mm height #2mm {#3.pdf}}}{\hbox{\pdfximage width #1mm height #2mm {#3.png}}}%
\pdfrefximage\pdflastximage\or%pdf
\IfFileExists{#3.bmp}{\includegraphics[bb=0 0 #1mm #2mm]{#3.bmp}}{\includegraphics[bb=0 0 #1mm #2mm]{#3.eps}}\or%ps
\hbox{\pdfimage width #1mm height #2mm {#3.png}}%pdfmi
\fi}

\def\tabcaption{\@ifnextchar*{\tabcaption@short}{\tabcaption@full}}

\def\tabcaption@full#1#2#3{
\parbox{#2mm}{\centering \hfill Т\,а\,б\,л\,и\,ц\,а\, #1\\[2mm] #3}\vspace*{3mm}
}

\def\tabcaption@short#1#2#3{
\parbox{#2mm}{\centering #3}\vspace*{3mm}
}

\def\tabcaptionend#1#2{
\parbox{#2mm}{\centering \hfill Т\,а\,б\,л\,и\,ц\,а\, #1 \,
(\,о\,к\,о\,н\,ч\,а\,н\,и\,е\,)}\vspace*{3mm}
}

\def\tabcaptioncont#1#2{
\parbox{#2mm}{\centering \hfill Т\,а\,б\,л\,и\,ц\,а\, #1 \,
(\,п\,р\,о\,д\,о\,л\,ж\,е\,н\,и\,е\,)}\vspace*{3mm}
}

\def\caption#1#2#3#4#5{\leavevmode
  \if0#1
     \def\risc{}\else\def\risc{{\it Рис.~#1.}}
  \fi
  \if#2c
    \parbox{#3mm}{\rule{#3mm}{.4pt}\\[2mm]\small
    \if#5**
      \risc~#4\\[.5mm] \rule{#3mm}{.4pt}
    \else
      \risc~#4\\%[1mm]
      \parbox[t]{#3mm}{\footnotesize#5\\[.8mm]\rule{#3mm}{.4pt}}
    \fi}
  \else
    \if#2b
      \parbox[t]{#3mm}{\small
      \if#5**
        \rule{0mm}{3mm}\if0#1#4\else\risc~#4\fi
      \else
        \rule{0mm}{3mm}\if0#1#4\else\risc~#4\fi\\%[1mm]
        \parbox[t]{#3mm}{\footnotesize#5}
      \fi}
    \else
      \parbox{#3mm}{\small\centerline{\rule{0mm}{3mm}{\it Рис.~#1}}}
    \fi
  \fi
}

\def\rb{\linebreak\null\hfil}

\newenvironment{proof}{\par\smallskip\pagebreak[2]%
\@ifnextchar[{\the@proof@named}{\the@proof}}
{\hfill$\Box$\par\smallskip}

\def\the@proof{\textbf{Доказательство.} }
\def\the@proof@named[#1]{\textbf{Доказательство #1.} }

\newenvironment{theorem}[1]{\par\smallskip\pagebreak[2]%
\textbf{Теорема #1.}\it}
{\par\smallskip}

\newenvironment{theorem*}{\par\smallskip\pagebreak[2]%
\textbf{Теорема.}\it}
{\par\smallskip}

\newenvironment{lemma}[1]{\par\smallskip\pagebreak[2]%
\textbf{Лемма #1.}\it}
{\par\smallskip}

\newenvironment{lemma*}{\par\smallskip\pagebreak[2]%
\textbf{Лемма.}\it}
{\par\smallskip}

\newenvironment{remark}[1]{\par\smallskip\pagebreak[2]%
\textbf{Замечание #1.}}
{\par\smallskip}

\newenvironment{remark*}{\par\smallskip\pagebreak[2]%
\textbf{Замечание.}}
{\par\smallskip}

\newenvironment{definition}[1]{\par\smallskip\pagebreak[2]%
\textbf{Определение #1.}}
{\par\smallskip}

\newenvironment{definition*}{\par\smallskip\pagebreak[2]%
\textbf{Определение.}}
{\par\smallskip}

\newenvironment{assumption}[1]{\par\smallskip\pagebreak[2]%
\textbf{Предположение #1.}}
{\par\smallskip}

\newenvironment{assumption*}{\par\smallskip\pagebreak[2]%
\textbf{Предположение.}}
{\par\smallskip}

\newenvironment{problem}[1]{\par\smallskip\pagebreak[2]%
\textbf{Задача #1.}}
{\par\smallskip}

\newenvironment{problem*}{\par\smallskip\pagebreak[2]%
\textbf{Задача.}}
{\par\smallskip}

\newenvironment{invproblem}[1]{\par\smallskip\pagebreak[2]%
\textbf{Обратная задача #1.}}
{\par\smallskip}

\newenvironment{invproblem*}{\par\smallskip\pagebreak[2]%
\textbf{Обратная задача.}}
{\par\smallskip}

\newenvironment{dirproblem}[1]{\par\smallskip\pagebreak[2]%
\textbf{Прямая задача #1.}}
{\par\smallskip}

\newenvironment{dirproblem*}{\par\smallskip\pagebreak[2]%
\textbf{Прямая задача.}}
{\par\smallskip}

\newenvironment{boundproblem}[1]{\par\smallskip\pagebreak[2]%
\textbf{Краевая задача #1.}}
{\par\smallskip}

\newenvironment{boundproblem*}{\par\smallskip\pagebreak[2]%
\textbf{Краевая задача.}}
{\par\smallskip}

\newenvironment{proclaim}[2]{\par\smallskip\pagebreak[2]%
\textbf{#1 #2.}}
{\par\smallskip}

\newenvironment{proclaim*}[1]{\par\smallskip\pagebreak[2]%
\textbf{#1.}}
{\par\smallskip}

\newenvironment{proclaimit}[2]{\par\smallskip\pagebreak[2]%
\textbf{#1 #2.}\it}
{\par\smallskip}

\newenvironment{proclaimit*}[1]{\par\smallskip\pagebreak[2]%
\textbf{#1.}\it}
{\par\smallskip}


\def\e{\mathop{{}\rm e}\nolimits}
\def\tr{\mathop{\rm tr}\nolimits}
\def\M{\mathop{{}\rm M}\nolimits}
\def\const{\mathop{\rm const}\nolimits}
\def\sgn{\mathop{\rm sgn}\nolimits}
\def\Re{\mathop{\rm Re}\nolimits}
\def\diver{\mathop{\rm div}\nolimits}
\def\rot{\mathop{\rm rot}\nolimits}
\def\grad{\mathop{\rm grad}\nolimits}
\def\arcthh{\mathop{\rm arcth}\nolimits}
\def\arthh{\mathop{\rm arth}\nolimits}

\def\No{\textnumero}

\def\map{\!:}

\newcommand{\?}{\textbf{?}\marginpar{\Large\textbf{?}}}

\let\leq\leqslant
\let\geq\geqslant
\let\le\leqslant
\let\ge\geqslant

\outformat=0
\issue{2020}{99}{9}
\pages{1}{10}
\doi{01}
\dates{дата1}{дата2}{дата3}

\let\caption\oldcaption