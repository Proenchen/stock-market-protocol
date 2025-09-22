from typing import List
from logic.analyzers.base import AnalyzerOutput

def compose_document(title: str, outputs: List[AnalyzerOutput]) -> str:
    # Nutze deinen existierenden LaTeX-Rahmen, aber füge die Blöcke dynamisch ein.
    parts = []
    for out in outputs:
        # Jeder Output kann mehrere Blöcke mitbringen:
        parts.append(f"\\section{{{out.name}}}\n" + "\n".join(out.latex_blocks))
    # Reuse deines vorhandenen Headers/Footers (kürze hier, nur Idee):
    return r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{float}
\title{""" + title + r"""}
\date{\today}
\begin{document}
\maketitle
""" + "\n\n".join(parts) + "\n\\end{document}\n"
